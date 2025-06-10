using System.Text.Json.Serialization;

namespace Inference.Data;

// See https://platform.openai.com/docs/api-reference/chat/create for documentation.

// Shared types
[JsonConverter(typeof(JsonStringEnumConverter))]
public enum Role
{
    [JsonStringEnumMemberName("system")]  System,
    [JsonStringEnumMemberName("user")]  User,
    [JsonStringEnumMemberName("assistant")]  Assistant
}

public record Message(
    [property: JsonPropertyName("role")] Role Role,
    [property: JsonPropertyName("content")] string Content
);

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum Model
{
    [JsonStringEnumMemberName("google/gemma-3-12b")] Gemma_3_12b,
    [JsonStringEnumMemberName("google/gemma-3-27b")] Gemma_3_27b,
    [JsonStringEnumMemberName("qwen/qwen3-1.7b")] Qwen_3_1_7b,
    [JsonStringEnumMemberName("qwen/qwen3-4b")] Qwen_3_4b,
    [JsonStringEnumMemberName("qwen/qwen3-32b")] Qwen_3_32b,
    [JsonStringEnumMemberName("deepseek/deepseek-r1-0528-qwen3-8b")] DeepSeek_R1_Qwen3_8b,
    [JsonStringEnumMemberName("microsoft/phi-4")] MS_Phi_4,
    [JsonStringEnumMemberName("microsoft/phi-4-reasoning-plus")] MS_Phi_4_Reasoning_Plus,
    [JsonStringEnumMemberName("ibm/granite-3.2-8b")] IBM_Granite_3_2_28,
}

// Request
public record RequestBody(
    [property: JsonPropertyName("model")] Model Model,
    [property: JsonPropertyName("messages")] IReadOnlyList<Message> Messages,
    [property: JsonPropertyName("temperature")] double Temperature,
    [property: JsonPropertyName("max_tokens")] int MaxTokens,
    [property: JsonPropertyName("stream")] bool Stream
);

// Response
public record Choice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("logprobs")] object Logprobs,
    [property: JsonPropertyName("finish_reason")] string FinishReason,
    [property: JsonPropertyName("message")] Message Message
);

public record ResponseBody(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("created")] int Created,
    [property: JsonPropertyName("model")] Model Model,
    [property: JsonPropertyName("choices")] IReadOnlyList<Choice> Choices,
    [property: JsonPropertyName("usage")] Usage Usage,
    [property: JsonPropertyName("stats")] Stats Stats,
    [property: JsonPropertyName("system_fingerprint")] string SystemFingerprint
);

public record Stats;

public record Usage(
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("completion_tokens")] int CompletionTokens,
    [property: JsonPropertyName("total_tokens")] int TotalTokens
);