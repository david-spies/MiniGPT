// ContentView.swift
// MiniGPT iOS — SwiftUI Interface

import SwiftUI

struct ContentView: View {
    @StateObject private var engine = MiniGPTEngine()
    @State private var prompt = "Once upon a time, there was a tiny robot who"
    @State private var maxTokens = 80.0
    @State private var temperature = 0.8
    @State private var topK = 40.0
    @State private var modelLoaded = false
    @State private var loadError: String?

    var body: some View {
        NavigationStack {
            ZStack {
                Color(red: 0.04, green: 0.04, blue: 0.07)
                    .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 16) {

                        // ── Header ────────────────────────────────────────
                        HStack {
                            Label("MiniGPT", systemImage: "bolt.fill")
                                .font(.system(size: 22, weight: .bold, design: .monospaced))
                                .foregroundStyle(.white)
                            Spacer()
                            Text("~1.5MB")
                                .font(.system(size: 11, design: .monospaced))
                                .padding(.horizontal, 8).padding(.vertical, 4)
                                .background(Color.green.opacity(0.15))
                                .foregroundStyle(.green)
                                .clipShape(Capsule())
                        }
                        .padding(.horizontal)
                        .padding(.top, 8)

                        // ── Load / Stats ──────────────────────────────────
                        if !modelLoaded {
                            loadCard
                        } else {
                            statsRow
                        }

                        // ── Prompt ────────────────────────────────────────
                        VStack(alignment: .leading, spacing: 8) {
                            Text("PROMPT")
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                .foregroundStyle(Color(white: 0.45))
                            TextEditor(text: $prompt)
                                .font(.system(size: 14, design: .monospaced))
                                .frame(minHeight: 90)
                                .padding(10)
                                .background(Color(white: 0.1))
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .foregroundStyle(.white)
                        }
                        .padding(.horizontal)

                        // ── Controls ──────────────────────────────────────
                        VStack(spacing: 12) {
                            SliderRow(label: "MAX TOKENS", value: $maxTokens, range: 20...200, display: { "\(Int($0))" })
                            SliderRow(label: "TEMPERATURE", value: $temperature, range: 0.1...1.5, display: { String(format: "%.1f", $0) })
                            SliderRow(label: "TOP-K", value: $topK, range: 1...100, display: { "\(Int($0))" })
                        }
                        .padding(.horizontal)

                        // ── Action Buttons ────────────────────────────────
                        HStack(spacing: 10) {
                            Button {
                                Task { await engine.generate(
                                    prompt: prompt,
                                    maxNewTokens: Int(maxTokens),
                                    temperature: Float(temperature),
                                    topK: Int(topK)
                                )}
                            } label: {
                                Label("Generate", systemImage: "play.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .font(.system(size: 15, weight: .semibold))
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(Color(red: 0.36, green: 0.43, blue: 0.96))
                            .disabled(!modelLoaded || engine.isGenerating)

                            if engine.isGenerating {
                                Button {
                                    engine.cancel()
                                } label: {
                                    Label("Stop", systemImage: "stop.fill")
                                        .padding(.vertical, 12)
                                        .padding(.horizontal, 16)
                                        .font(.system(size: 15, weight: .semibold))
                                }
                                .buttonStyle(.bordered)
                                .foregroundStyle(.red)
                            }
                        }
                        .padding(.horizontal)

                        // ── Output ────────────────────────────────────────
                        VStack(alignment: .leading, spacing: 8) {
                            Text("OUTPUT")
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                .foregroundStyle(Color(white: 0.45))
                            ScrollView {
                                Text(engine.outputText.isEmpty ? "Output will appear here…" : engine.outputText)
                                    .font(.system(size: 14, design: .monospaced))
                                    .foregroundStyle(engine.outputText.isEmpty ? Color(white: 0.3) : .white)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(14)
                            }
                            .frame(minHeight: 160)
                            .background(Color(white: 0.08))
                            .clipShape(RoundedRectangle(cornerRadius: 8))

                            if !engine.outputText.isEmpty {
                                ShareLink(item: engine.outputText) {
                                    Label("Save Story", systemImage: "square.and.arrow.down")
                                        .font(.system(size: 12, design: .monospaced))
                                        .foregroundStyle(Color(red: 0.36, green: 0.43, blue: 0.96))
                                }
                            }
                        }
                        .padding(.horizontal)
                        .padding(.bottom, 32)
                    }
                }
            }
            .navigationBarHidden(true)
        }
        .preferredColorScheme(.dark)
    }

    // ── Subviews ──────────────────────────────────────────────────────────

    var loadCard: some View {
        VStack(spacing: 12) {
            if let err = loadError {
                Text(err)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            }
            Button {
                Task {
                    do {
                        try await engine.load()
                        modelLoaded = true
                    } catch {
                        loadError = error.localizedDescription
                    }
                }
            } label: {
                Text("Load Model (~1.5MB)")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 13)
                    .font(.system(size: 15, weight: .semibold))
            }
            .buttonStyle(.borderedProminent)
            .tint(Color(red: 0.36, green: 0.43, blue: 0.96))
        }
        .padding(.horizontal)
    }

    var statsRow: some View {
        HStack(spacing: 12) {
            StatPill(label: "Speed", value: engine.isGenerating ? "—" : "\(Int(engine.tokensPerSecond))", unit: "tok/s")
            StatPill(label: "Device", value: "ANE", unit: "Neural Engine")
            StatPill(label: "Size", value: "1.5", unit: "MB INT8")
        }
        .padding(.horizontal)
    }
}

// MARK: - Supporting Views

struct SliderRow: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let display: (Double) -> String

    var body: some View {
        HStack {
            Text(label)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(Color(white: 0.4))
                .frame(width: 100, alignment: .leading)
            Slider(value: $value, in: range)
                .tint(Color(red: 0.36, green: 0.43, blue: 0.96))
            Text(display(value))
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(Color(red: 0.36, green: 0.43, blue: 0.96))
                .frame(width: 36, alignment: .trailing)
        }
    }
}

struct StatPill: View {
    let label: String
    let value: String
    let unit: String

    var body: some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 18, weight: .bold, design: .monospaced))
                .foregroundStyle(Color(red: 0.36, green: 0.43, blue: 0.96))
            Text(label)
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(Color(white: 0.4))
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(Color(white: 0.07))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

#Preview {
    ContentView()
}
