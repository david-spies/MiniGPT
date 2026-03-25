// MiniGPTInference.swift
// MiniGPT iOS — CoreML on-device inference
// Convert model: python scripts/export_coreml.py
// Drop the generated .mlpackage into Xcode and add to target.

import Foundation
import CoreML
import Accelerate

// MARK: - Config

struct MiniGPTConfig {
    let vocabSize   = 5000
    let nEmbd       = 128
    let nHead       = 4
    let nLayer      = 4
    let blockSize   = 256
    let eosTokenId  = 2
    let bosTokenId  = 0
    let padTokenId  = 1
}

// MARK: - Tokenizer (loads vocab.json + merges.txt from bundle)

final class MiniGPTTokenizer {
    private var vocab: [String: Int] = [:]
    private var idToToken: [Int: String] = [:]

    init(bundleDirectory: String = "mini_gpt_tokenizer") {
        guard
            let vocabURL = Bundle.main.url(
                forResource: "vocab", withExtension: "json",
                subdirectory: bundleDirectory),
            let data = try? Data(contentsOf: vocabURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Int]
        else {
            print("[MiniGPT] Warning: Could not load tokenizer vocab.")
            return
        }
        self.vocab = json
        self.idToToken = Dictionary(uniqueKeysWithValues: json.map { ($1, $0) })
    }

    func encode(_ text: String) -> [Int] {
        var tokens = [MiniGPTConfig().bosTokenId]
        // Byte-level BPE requires the full merge table for correctness.
        // For demo, we use character-level with Ġ prefix for spaces.
        var processed = text
        for (i, char) in processed.enumerated() {
            let key = (i == 0 ? "" : char == " " ? "Ġ" : String(char))
            let id = vocab[key] ?? vocab[String(char)] ?? MiniGPTConfig().vocabSize - 1
            tokens.append(id)
        }
        return tokens
    }

    func decode(_ ids: [Int]) -> String {
        let cfg = MiniGPTConfig()
        return ids
            .filter { $0 != cfg.bosTokenId && $0 != cfg.eosTokenId }
            .compactMap { idToToken[$0] }
            .map { tok -> String in
                if tok.hasPrefix("Ġ") { return " " + tok.dropFirst() }
                return tok
            }
            .joined()
    }
}

// MARK: - Inference Engine

@MainActor
final class MiniGPTEngine: ObservableObject {

    @Published var outputText: String = ""
    @Published var isGenerating: Bool = false
    @Published var tokensPerSecond: Double = 0

    private var model: MLModel?
    private let tokenizer = MiniGPTTokenizer()
    private let config    = MiniGPTConfig()
    private var cancelFlag = false

    // ── Load ──────────────────────────────────────────────────────────────
    func load() async throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine  // Prefer Apple Neural Engine
        cfg.allowLowPrecisionAccumulationOnGPU = true

        guard let url = Bundle.main.url(forResource: "mini_gpt", withExtension: "mlpackage") else {
            throw MiniGPTError.modelNotFound
        }
        model = try await MLModel.load(contentsOf: url, configuration: cfg)
        print("[MiniGPT] Model loaded on Neural Engine.")
    }

    // ── Generate ──────────────────────────────────────────────────────────
    func generate(
        prompt: String,
        maxNewTokens: Int = 100,
        temperature: Float = 0.8,
        topK: Int = 40
    ) async {
        guard let model = model else { return }

        isGenerating  = true
        cancelFlag    = false
        outputText    = prompt

        var tokenIds  = tokenizer.encode(prompt)
        let startTime = Date()
        var generated = 0

        // Process full prompt first
        do {
            var (logits, pastKV) = try runForward(model: model, ids: tokenIds, pastKV: nil)

            for _ in 0 ..< maxNewTokens {
                if cancelFlag { break }

                let nextId = sampleTopK(logits: logits, k: topK, temperature: temperature)
                if nextId == config.eosTokenId { break }

                let piece = tokenizer.decode([nextId])
                outputText += piece
                generated  += 1
                tokenIds.append(nextId)

                // Incremental decode: single token
                (logits, pastKV) = try runForward(model: model, ids: [nextId], pastKV: pastKV)
            }
        } catch {
            print("[MiniGPT] Inference error: \(error)")
        }

        let elapsed = Date().timeIntervalSince(startTime)
        tokensPerSecond = Double(generated) / elapsed
        isGenerating = false
    }

    func cancel() { cancelFlag = true }

    // ── Forward Pass ──────────────────────────────────────────────────────
    private func runForward(
        model: MLModel,
        ids: [Int],
        pastKV: MLFeatureProvider?
    ) throws -> ([Float], MLFeatureProvider?) {
        // Build input feature dict
        let inputArray = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        for (i, id) in ids.enumerated() {
            inputArray[i] = NSNumber(value: id)
        }

        var featureDict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputArray)
        ]

        // Attach past KV cache if available
        if let past = pastKV {
            for name in past.featureNames {
                if let val = past.featureValue(for: name) {
                    featureDict[name] = val
                }
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: featureDict)
        let output   = try model.prediction(from: provider)

        // Extract logits
        guard let logitArray = output.featureValue(for: "logits")?.multiArrayValue else {
            throw MiniGPTError.invalidOutput
        }

        let seqLen    = logitArray.shape[1].intValue
        let vocabSize = logitArray.shape[2].intValue
        let offset    = (seqLen - 1) * vocabSize
        var logits    = [Float](repeating: 0, count: vocabSize)
        for i in 0 ..< vocabSize {
            logits[i] = logitArray[offset + i].floatValue
        }

        return (logits, output)
    }

    // ── Sampling ──────────────────────────────────────────────────────────
    private func sampleTopK(logits: [Float], k: Int, temperature: Float) -> Int {
        var scaled = logits.map { $0 / temperature }

        // Top-k filtering
        let sorted = scaled.enumerated().sorted { $0.element > $1.element }
        let threshold = sorted[min(k - 1, sorted.count - 1)].element
        scaled = scaled.map { $0 < threshold ? -Float.infinity : $0 }

        // Softmax
        let maxVal = scaled.max() ?? 0
        var exps   = scaled.map { exp($0 - maxVal) }
        let sum    = exps.reduce(0, +)
        exps = exps.map { $0 / sum }

        // Multinomial sample
        var r = Float.random(in: 0 ..< 1)
        for (i, p) in exps.enumerated() {
            r -= p
            if r <= 0 { return i }
        }
        return sorted[0].offset
    }
}

// MARK: - Errors

enum MiniGPTError: Error, LocalizedError {
    case modelNotFound
    case invalidOutput

    var errorDescription: String? {
        switch self {
        case .modelNotFound: return "mini_gpt.mlpackage not found in app bundle."
        case .invalidOutput: return "Model returned unexpected output format."
        }
    }
}
