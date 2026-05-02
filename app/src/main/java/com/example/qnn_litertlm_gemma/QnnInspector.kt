package com.example.qnn_litertlm_gemma

import com.qualcomm.qti.QnnDelegate
import timber.log.Timber

object QnnInspector {
    private const val TAG = "QnnInspector"

    fun inspect() {
        Timber.tag(TAG).d("--- QnnDelegate Methods ---")
        QnnDelegate::class.java.methods.forEach {
            Timber.tag(TAG).d("Method: ${it.name}(${it.parameterTypes.joinToString { p -> p.simpleName }}) -> ${it.returnType.simpleName}")
        }
        QnnDelegate::class.java.constructors.forEach {
            Timber.tag(TAG).d("Constructor: ${it.parameterTypes.joinToString { p -> p.simpleName }}")
        }

        Timber.tag(TAG).d("--- QnnDelegate.Options Methods ---")
        QnnDelegate.Options::class.java.methods.forEach {
            Timber.tag(TAG).d("Method: ${it.name}(${it.parameterTypes.joinToString { p -> p.simpleName }}) -> ${it.returnType.simpleName}")
        }
    }
}
