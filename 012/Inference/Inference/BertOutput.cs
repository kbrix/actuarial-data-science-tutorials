using Microsoft.ML.Data;

namespace Inference;

/// For the model 'distilbert-base-multilingual-cased.onnx', see 'https://netron.app' to inspect the model.
public class BertOutput
{
    // [ColumnName("logits")] // // Matches 'input_ids' from the model input names
    // public float[] Logits { get; set; }
    [ColumnName("last_hidden_state")] // // Matches 'last_hidden_state' from the model input names
    public float[] LastHiddenState { get; set; }
}