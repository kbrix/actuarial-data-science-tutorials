using Newtonsoft.Json;

namespace Inference.Data;

public record Choice(
    [property: JsonProperty("index")] int Index,
    [property: JsonProperty("logprobs")] object Logprobs,
    [property: JsonProperty("finish_reason")] string FinishReason,
    [property: JsonProperty("message")] Message Message
);

public record ResponseBody(
    [property: JsonProperty("id")] string Id,
    [property: JsonProperty("object")] string Object,
    [property: JsonProperty("created")] int Created,
    [property: JsonProperty("model")] string Model,
    [property: JsonProperty("choices")] IReadOnlyList<Choice> Choices,
    [property: JsonProperty("usage")] Usage Usage,
    [property: JsonProperty("stats")] Stats Stats,
    [property: JsonProperty("system_fingerprint")] string SystemFingerprint
);

public record Stats(

);

public record Usage(
    [property: JsonProperty("prompt_tokens")] int PromptTokens,
    [property: JsonProperty("completion_tokens")] int CompletionTokens,
    [property: JsonProperty("total_tokens")] int TotalTokens
);