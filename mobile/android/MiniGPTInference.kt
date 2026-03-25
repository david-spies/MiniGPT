// MiniGPTInference.kt
// MiniGPT Android — TensorFlow Lite on-device inference
// Convert model: python scripts/export_tflite.py
// Place mini_gpt.tflite + tokenizer/vocab.json in app/src/main/assets/

package com.minigpt.inference

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.ln

private const val TAG = "MiniGPT"

// ── Config ─────────────────────────────────────────────────────────────────

data class MiniGPTConfig(
    val vocabSize: Int   = 5000,
    val nEmbd: Int       = 128,
    val nHead: Int       = 4,
    val nLayer: Int      = 4,
    val blockSize: Int   = 256,
    val bosTokenId: Int  = 0,
    val padTokenId: Int  = 1,
    val eosTokenId: Int  = 2,
    val unkTokenId: Int  = 3,
) {
    val headDim: Int get() = nEmbd / nHead
}

// ── Tokenizer ──────────────────────────────────────────────────────────────

class MiniGPTTokenizer(context: Context) {
    private val vocab = mutableMapOf<String, Int>()
    private val idToToken = mutableMapOf<Int, String>()
    private val cfg = MiniGPTConfig()

    init {
        runCatching {
            val json = context.assets.open("mini_gpt_tokenizer/vocab.json")
                .bufferedReader().readText()
            val obj = JSONObject(json)
            obj.keys().forEach { key ->
                val id = obj.getInt(key)
                vocab[key] = id
                idToToken[id] = key
            }
        }.onFailure { Log.w(TAG, "Tokenizer load failed: ${it.message}") }
    }

    fun encode(text: String): IntArray {
        val tokens = mutableListOf(cfg.bosTokenId)
        text.forEachIndexed { i, char ->
            val key = if (i > 0 && char == ' ') "Ġ${text[i + 1]}"
                      else char.toString()
            tokens.add(vocab[key] ?: cfg.unkTokenId)
        }
        return tokens.toIntArray()
    }

    fun decode(ids: IntArray): String = ids
        .filter { it != cfg.bosTokenId && it != cfg.eosTokenId }
        .joinToString("") { id ->
            val tok = idToToken[id] ?: ""
            if (tok.startsWith("Ġ")) " ${tok.drop(1)}" else tok
        }
}

// ── Inference Engine ────────────────────────────────────────────────────────

class MiniGPTEngine(private val context: Context) {

    private var interpreter: Interpreter? = null
    private val tokenizer = MiniGPTTokenizer(context)
    private val cfg = MiniGPTConfig()
    private var cancelFlag = false

    // ── Load ───────────────────────────────────────────────────────────────
    suspend fun load() = withContext(Dispatchers.IO) {
        val opts = Interpreter.Options().apply {
            numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
            // Use GPU delegate if available, fall back to CPU NNAPI
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                addDelegate(GpuDelegate(compatList.bestOptionsForThisDevice))
                Log.i(TAG, "GPU delegate enabled")
            } else {
                useNNAPI = true
                Log.i(TAG, "NNAPI delegate enabled")
            }
        }
        interpreter = Interpreter(loadModelFile(), opts)
        Log.i(TAG, "Model loaded successfully")
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fd = context.assets.openFd("mini_gpt.tflite")
        return FileInputStream(fd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength
        )
    }

    // ── Generate (streaming Flow) ──────────────────────────────────────────
    fun generate(
        prompt: String,
        maxNewTokens: Int = 100,
        temperature: Float = 0.8f,
        topK: Int = 40,
    ): Flow<String> = flow {
        cancelFlag = false
        val startTime = System.currentTimeMillis()
        var generated = 0

        val tokenIds = tokenizer.encode(prompt).toMutableList()

        // Process full prompt
        var (logits, pastKV) = runForward(tokenIds.toIntArray(), null)

        repeat(maxNewTokens) {
            if (cancelFlag) return@flow

            val nextId = sampleTopK(logits, topK, temperature)
            if (nextId == cfg.eosTokenId) return@flow

            val piece = tokenizer.decode(intArrayOf(nextId))
            emit(piece)
            generated++
            tokenIds.add(nextId)

            val result = runForward(intArrayOf(nextId), pastKV)
            logits = result.first
            pastKV = result.second
        }

        val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
        val tps = generated / elapsed
        Log.i(TAG, "Generation done: $generated tokens @ %.1f tok/s".format(tps))
    }.flowOn(Dispatchers.Default)

    fun cancel() { cancelFlag = true }

    // ── Forward ────────────────────────────────────────────────────────────
    private fun runForward(
        ids: IntArray,
        pastKV: Array<ByteBuffer>?,
    ): Pair<FloatArray, Array<ByteBuffer>> {
        val interp = interpreter ?: error("Model not loaded")

        // Input tensor: [1, seq_len]
        val inputBuf = ByteBuffer.allocateDirect(ids.size * 4)
            .order(ByteOrder.nativeOrder())
        ids.forEach { inputBuf.putInt(it) }
        inputBuf.rewind()

        // Build inputs map
        val inputs = mutableMapOf<String, Any>("input_ids" to inputBuf)
        if (pastKV != null) {
            for (i in 0 until cfg.nLayer) {
                inputs["past_k_$i"] = pastKV[i * 2]
                inputs["past_v_$i"] = pastKV[i * 2 + 1]
            }
        } else {
            // Empty caches
            for (i in 0 until cfg.nLayer) {
                val empty = ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder())
                inputs["past_k_$i"] = empty
                inputs["past_v_$i"] = empty
            }
        }

        // Output buffers
        val logitsBuf = ByteBuffer.allocateDirect(cfg.vocabSize * 4)
            .order(ByteOrder.nativeOrder())
        val newKVBufs = Array(cfg.nLayer * 2) {
            ByteBuffer.allocateDirect(cfg.nHead * cfg.headDim * 4 * (ids.size + (pastKV?.size ?: 0)))
                .order(ByteOrder.nativeOrder())
        }

        val outputs = mutableMapOf<String, Any>("logits" to logitsBuf)
        for (i in 0 until cfg.nLayer) {
            outputs["present_k_$i"] = newKVBufs[i * 2]
            outputs["present_v_$i"] = newKVBufs[i * 2 + 1]
        }

        interp.runForMultipleInputsOutputs(inputs.values.toTypedArray(), outputs)

        // Extract last token logits
        logitsBuf.rewind()
        val logits = FloatArray(cfg.vocabSize) { logitsBuf.float }

        newKVBufs.forEach { it.rewind() }
        return Pair(logits, newKVBufs)
    }

    // ── Top-K Sampling ────────────────────────────────────────────────────
    private fun sampleTopK(logits: FloatArray, k: Int, temperature: Float): Int {
        val scaled = logits.map { it / temperature }.toFloatArray()

        // Find top-k threshold
        val sorted = scaled.indices.sortedByDescending { scaled[it] }
        val threshold = scaled[sorted[minOf(k - 1, sorted.size - 1)]]

        // Softmax over top-k
        val maxVal = scaled.max()!!
        var sum = 0.0
        val exps = FloatArray(scaled.size) { i ->
            if (scaled[i] >= threshold) {
                exp((scaled[i] - maxVal).toDouble()).toFloat().also { sum += it }
            } else 0f
        }
        val probs = exps.map { it / sum.toFloat() }

        // Multinomial sample
        var r = Math.random()
        for ((i, p) in probs.withIndex()) {
            r -= p
            if (r <= 0.0) return i
        }
        return sorted[0]
    }
}
