using System.Runtime.Serialization;
using Newtonsoft.Json;

namespace Inference.Data;

// See https://platform.openai.com/docs/api-reference/chat/create for documentation.

public enum Role
{
    [EnumMember(Value = "system")]  System,
    [EnumMember(Value = "user")]  User
}

public record Message(
    [property: JsonProperty("role")] Role Role,
    [property: JsonProperty("content")] string Content
);

public enum Model
{
    [EnumMember(Value = "google/gemma-3-12b")] Gemma_3_12b
}

public record RequestBody(
    [property: JsonProperty("model")] Model Model,
    [property: JsonProperty("messages")] IReadOnlyList<Message> Messages,
    [property: JsonProperty("temperature")] double Temperature,
    [property: JsonProperty("max_tokens")] int MaxTokens,
    [property: JsonProperty("stream")] bool Stream
);