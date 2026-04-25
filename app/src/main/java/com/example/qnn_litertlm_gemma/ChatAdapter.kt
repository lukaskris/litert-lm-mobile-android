package com.example.qnn_litertlm_gemma

import android.graphics.BitmapFactory
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.lifecycle.findViewTreeLifecycleOwner
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.qnn_litertlm_gemma.databinding.ItemMessageAssistantBinding
import com.example.qnn_litertlm_gemma.databinding.ItemMessageUserBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * RecyclerView adapter for chat messages
 */
class ChatAdapter : ListAdapter<ChatMessage, RecyclerView.ViewHolder>(ChatMessageDiffCallback()) {

    companion object {
        private const val VIEW_TYPE_USER = 1
        private const val VIEW_TYPE_ASSISTANT = 2
        const val PAYLOAD_STREAMING_UPDATE = "streaming_update"
        const val PAYLOAD_THINKING_UPDATE = "thinking_update"
        private val timeFormat = SimpleDateFormat("HH:mm", Locale.getDefault())
    }

    override fun getItemViewType(position: Int): Int {
        return when (getItem(position).sender) {
            MessageSender.USER -> VIEW_TYPE_USER
            MessageSender.ASSISTANT, MessageSender.SYSTEM -> VIEW_TYPE_ASSISTANT
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return when (viewType) {
            VIEW_TYPE_USER -> {
                val binding = ItemMessageUserBinding.inflate(
                    LayoutInflater.from(parent.context), parent, false
                )
                UserMessageViewHolder(binding)
            }
            else -> {
                val binding = ItemMessageAssistantBinding.inflate(
                    LayoutInflater.from(parent.context), parent, false
                )
                AssistantMessageViewHolder(binding)
            }
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = getItem(position)
        when (holder) {
            is UserMessageViewHolder -> holder.bind(message)
            is AssistantMessageViewHolder -> holder.bind(message)
        }
    }

    override fun onViewRecycled(holder: RecyclerView.ViewHolder) {
        super.onViewRecycled(holder)
        if (holder is UserMessageViewHolder) {
            holder.cancelImageLoad()
        }
    }

    /**
     * Partial update via payloads — only update text + indicators
     * without a full rebind (no re-measure / re-layout needed).
     */
    override fun onBindViewHolder(
        holder: RecyclerView.ViewHolder,
        position: Int,
        payloads: MutableList<Any>
    ) {
        if (payloads.isEmpty()) {
            super.onBindViewHolder(holder, position, payloads)
            return
        }

        val message = getItem(position)
        when (holder) {
            is UserMessageViewHolder -> holder.bind(message)
            is AssistantMessageViewHolder -> {
                for (payload in payloads) {
                    when (payload) {
                        PAYLOAD_STREAMING_UPDATE -> {
                            holder.binding.textMessage.text = message.content
                            holder.updateThinkingAndTimestamp(message)
                        }
                        PAYLOAD_THINKING_UPDATE -> {
                            holder.updateThinkingAndTimestamp(message)
                        }
                    }
                }
            }
        }
    }

    inner class UserMessageViewHolder(
        internal val binding: ItemMessageUserBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        private var imageLoadJob: Job? = null

        fun bind(message: ChatMessage) {
            // Cancel any previous image load for this view
            imageLoadJob?.cancel()
            imageLoadJob = null

            binding.textMessage.text = message.content
            binding.textTimestamp.text = timeFormat.format(message.timestamp)

            // Reset image state immediately
            binding.imageAttached.setImageDrawable(null)
            binding.imageAttached.visibility = View.GONE

            // Load image asynchronously — never block the main thread
            if (message.imagePath != null) {
                val file = File(message.imagePath)
                if (file.exists()) {
                    val scope = itemView.findViewTreeLifecycleOwner()?.let {
                        CoroutineScope(Dispatchers.Main.immediate + Job())
                    } ?: return // No lifecycle — skip image

                    imageLoadJob = scope.launch {
                        val bitmap = withContext(Dispatchers.IO) {
                            BitmapFactory.decodeFile(file.absolutePath)
                        }
                        // Only set if the job hasn't been cancelled
                        ensureActive()
                        if (bitmap != null) {
                            binding.imageAttached.setImageBitmap(bitmap)
                            binding.imageAttached.visibility = View.VISIBLE
                            binding.textMessage.setPadding(
                                binding.textMessage.paddingStart,
                                8,
                                binding.textMessage.paddingEnd,
                                binding.textMessage.paddingBottom
                            )
                        }
                    }
                }
            }
        }

        fun cancelImageLoad() {
            imageLoadJob?.cancel()
            imageLoadJob = null
        }
    }

    inner class AssistantMessageViewHolder(
        internal val binding: ItemMessageAssistantBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(message: ChatMessage) {
            binding.textMessage.text = message.content
            updateThinkingAndTimestamp(message)
        }

        fun updateThinkingAndTimestamp(message: ChatMessage) {
            // Thinking status: show while streaming and no content yet
            if (message.isStreaming && message.content.isEmpty() && message.thinkingSeconds > 0) {
                binding.textThinkingStatus.text = "${message.thinkingLabel} ${message.thinkingSeconds}s"
                binding.textThinkingStatus.visibility = View.VISIBLE
                binding.textTimestamp.visibility = View.GONE
            } else {
                binding.textThinkingStatus.visibility = View.GONE
                binding.textTimestamp.visibility = View.VISIBLE
                binding.textTimestamp.text = timeFormat.format(message.timestamp)
            }
        }
    }
}

/**
 * DiffUtil callback for efficient list updates
 */
class ChatMessageDiffCallback : DiffUtil.ItemCallback<ChatMessage>() {
    override fun areItemsTheSame(oldItem: ChatMessage, newItem: ChatMessage): Boolean {
        return oldItem.timestamp == newItem.timestamp && oldItem.sender == newItem.sender
    }

    override fun areContentsTheSame(oldItem: ChatMessage, newItem: ChatMessage): Boolean {
        return oldItem == newItem
    }
}
