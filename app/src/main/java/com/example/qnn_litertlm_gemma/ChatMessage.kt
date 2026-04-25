package com.example.qnn_litertlm_gemma

import java.util.Date

/**
 * Represents a chat message in the conversation
 */
data class ChatMessage(
    val sender: MessageSender,
    val content: String,
    val timestamp: Date = Date(),
    val isStreaming: Boolean = false,
    val imagePath: String? = null,
    val thinkingSeconds: Int = 0,
    val thinkingLabel: String = "Thinking..."
)

/**
 * Enum for message sender types
 */
enum class MessageSender {
    USER,
    ASSISTANT,
    SYSTEM
}
