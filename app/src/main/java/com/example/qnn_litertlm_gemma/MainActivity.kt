package com.example.qnn_litertlm_gemma

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.example.qnn_litertlm_gemma.ocr.presentation.OcrDigitalApp
import com.example.qnn_litertlm_gemma.ocr.presentation.OcrDigitalViewModel
import com.example.qnn_litertlm_gemma.ui.theme.OcrDigitalTheme

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val viewModel = androidx.lifecycle.ViewModelProvider(this)[OcrDigitalViewModel::class.java]

        setContent {
            OcrDigitalTheme {
                OcrDigitalApp(viewModel)
            }
        }
    }
}
