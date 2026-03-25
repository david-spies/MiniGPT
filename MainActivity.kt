// MainActivity.kt
// MiniGPT Android — Jetpack Compose UI

package com.minigpt.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.minigpt.inference.MiniGPTEngine
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

// ── Color Palette ─────────────────────────────────────────────────────────────

private val Bg0    = Color(0xFF0A0B0E)
private val Bg1    = Color(0xFF10121A)
private val Bg2    = Color(0xFF161924)
private val Border = Color(0xFF252A3D)
private val Accent = Color(0xFF5B6EF5)
private val Green  = Color(0xFF34D399)
private val Text0  = Color(0xFFE8EAF2)
private val Text1  = Color(0xFF9BA3C2)
private val Text2  = Color(0xFF5A6180)

// ── ViewModel ─────────────────────────────────────────────────────────────────

class MiniGPTViewModel : ViewModel() {
    val engine by lazy { MiniGPTEngine(app) }

    // Inject Application context — in production use Hilt or a factory
    private lateinit var app: android.app.Application

    var modelLoaded by mutableStateOf(false)
    var isGenerating by mutableStateOf(false)
    var output by mutableStateOf("")
    var tokensPerSecond by mutableStateOf(0.0)
    var loadError by mutableStateOf<String?>(null)
    var statusMessage by mutableStateOf("Tap 'Load Model' to begin")

    fun init(application: android.app.Application) {
        app = application
    }

    fun loadModel() {
        viewModelScope.launch {
            statusMessage = "Loading model…"
            runCatching { engine.load() }
                .onSuccess {
                    modelLoaded = true
                    statusMessage = "Model ready"
                }
                .onFailure {
                    loadError = it.message
                    statusMessage = "Load failed"
                }
        }
    }

    fun generate(prompt: String, maxTokens: Int, temperature: Float, topK: Int) {
        if (!modelLoaded || isGenerating) return
        isGenerating = true
        output = prompt
        statusMessage = "Generating…"

        viewModelScope.launch {
            engine.generate(prompt, maxTokens, temperature, topK)
                .catch { e -> statusMessage = "Error: ${e.message}" }
                .collect { piece -> output += piece }

            tokensPerSecond = engine.tokensPerSecond
            isGenerating = false
            statusMessage = "Done · ${tokensPerSecond.toInt()} tok/s"
        }
    }

    fun cancel() {
        engine.cancel()
        isGenerating = false
    }
}

// ── Main Activity ─────────────────────────────────────────────────────────────

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                MiniGPTApp(application)
            }
        }
    }
}

// ── Root Composable ───────────────────────────────────────────────────────────

@Composable
fun MiniGPTApp(
    application: android.app.Application,
    vm: MiniGPTViewModel = viewModel()
) {
    LaunchedEffect(Unit) { vm.init(application) }

    var prompt by remember { mutableStateOf("Once upon a time, there was a tiny robot who") }
    var maxTokens by remember { mutableFloatStateOf(80f) }
    var temperature by remember { mutableFloatStateOf(0.8f) }
    var topK by remember { mutableFloatStateOf(40f) }
    val scroll = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Bg0)
            .verticalScroll(scroll)
            .padding(16.dp)
    ) {
        // ── Header ────────────────────────────────────────────────────────
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(vertical = 12.dp)
        ) {
            Text(
                "⚡ MiniGPT",
                color = Text0,
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace
            )
            Spacer(Modifier.weight(1f))
            Chip("~1.5MB")
            Spacer(Modifier.width(6.dp))
            Chip("On-device", accent = true)
        }

        // ── Status bar ────────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(Bg2, RoundedCornerShape(8.dp))
                .padding(10.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(vm.statusMessage, color = Text1, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
            if (vm.tokensPerSecond > 0 && !vm.isGenerating) {
                Text("${vm.tokensPerSecond.toInt()} tok/s", color = Green, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
            }
        }

        Spacer(Modifier.height(16.dp))

        // ── Load / Controls ───────────────────────────────────────────────
        if (!vm.modelLoaded) {
            Button(
                onClick = { vm.loadModel() },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Accent)
            ) {
                Text("Load Model (~1.5MB)", fontFamily = FontFamily.Monospace)
            }
            vm.loadError?.let {
                Text(it, color = Color.Red, fontSize = 12.sp, modifier = Modifier.padding(top = 6.dp))
            }
        } else {
            // Prompt input
            SectionLabel("PROMPT")
            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                modifier = Modifier.fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Accent,
                    unfocusedBorderColor = Border,
                    focusedTextColor = Text0,
                    unfocusedTextColor = Text0,
                    cursorColor = Accent,
                    focusedContainerColor = Bg2,
                    unfocusedContainerColor = Bg2,
                ),
                textStyle = LocalTextStyle.current.copy(
                    fontFamily = FontFamily.Monospace, fontSize = 14.sp
                ),
                minLines = 3,
            )

            Spacer(Modifier.height(14.dp))

            // Sliders
            SliderRow("MAX TOKENS", maxTokens, 20f..200f) { maxTokens = it }
            SliderRow("TEMPERATURE", temperature, 0.1f..1.5f) { temperature = it }
            SliderRow("TOP-K", topK, 1f..100f) { topK = it }

            Spacer(Modifier.height(14.dp))

            // Buttons
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Button(
                    onClick = { vm.generate(prompt, maxTokens.toInt(), temperature, topK.toInt()) },
                    modifier = Modifier.weight(1f),
                    enabled = !vm.isGenerating,
                    colors = ButtonDefaults.buttonColors(containerColor = Accent),
                ) {
                    Icon(Icons.Default.PlayArrow, contentDescription = null)
                    Spacer(Modifier.width(4.dp))
                    Text("Generate", fontFamily = FontFamily.Monospace, fontWeight = FontWeight.SemiBold)
                }
                if (vm.isGenerating) {
                    OutlinedButton(
                        onClick = { vm.cancel() },
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.Red),
                    ) {
                        Icon(Icons.Default.Stop, contentDescription = null)
                        Spacer(Modifier.width(4.dp))
                        Text("Stop")
                    }
                }
            }

            Spacer(Modifier.height(16.dp))

            // Output
            SectionLabel("OUTPUT")
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .defaultMinSize(minHeight = 160.dp)
                    .background(Bg1, RoundedCornerShape(8.dp))
                    .padding(14.dp)
            ) {
                if (vm.output.isEmpty()) {
                    Text("Output appears here…", color = Text2, fontFamily = FontFamily.Monospace, fontSize = 14.sp)
                } else {
                    Text(vm.output, color = Text0, fontFamily = FontFamily.Monospace, fontSize = 14.sp)
                }
                if (vm.isGenerating) {
                    CircularProgressIndicator(
                        modifier = Modifier.align(Alignment.BottomEnd).size(18.dp),
                        strokeWidth = 2.dp, color = Accent
                    )
                }
            }
        }

        Spacer(Modifier.height(32.dp))
    }
}

// ── Reusable Components ────────────────────────────────────────────────────────

@Composable
fun SectionLabel(text: String) {
    Text(
        text, color = Text2, fontSize = 10.sp,
        fontWeight = FontWeight.Medium, fontFamily = FontFamily.Monospace,
        letterSpacing = 0.1.sp,
        modifier = Modifier.padding(bottom = 6.dp)
    )
}

@Composable
fun Chip(label: String, accent: Boolean = false) {
    val bg = if (accent) Accent.copy(alpha = 0.15f) else Bg2
    val fg = if (accent) Accent else Text2
    Box(
        Modifier.background(bg, RoundedCornerShape(100.dp)).padding(horizontal = 8.dp, vertical = 3.dp)
    ) {
        Text(label, color = fg, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
    }
}

@Composable
fun SliderRow(label: String, value: Float, range: ClosedFloatingPointRange<Float>, onchange: (Float) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 3.dp)) {
        Text(label, color = Text2, fontSize = 10.sp, fontFamily = FontFamily.Monospace, modifier = Modifier.width(100.dp))
        Slider(
            value = value, onValueChange = onchange, valueRange = range,
            modifier = Modifier.weight(1f),
            colors = SliderDefaults.colors(thumbColor = Accent, activeTrackColor = Accent)
        )
        Text(
            if (label == "TEMPERATURE") "%.1f".format(value) else value.toInt().toString(),
            color = Accent, fontSize = 12.sp, fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(36.dp), textAlign = androidx.compose.ui.text.style.TextAlign.End
        )
    }
}
