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
    [JsonStringEnumMemberName("google/gemma-3-12b")] Gemma_3_12b
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