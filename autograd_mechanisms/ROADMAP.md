## **Phase 2 – Core Neural Network Engine**

You need a way to build and train neural networks before you attempt an LLM.

1. **Automatic Differentiation (Autograd)**

   * Implement a computation graph that records operations during forward pass.
   * Implement backward pass by traversing the graph in reverse.
   * Support at least: `matmul`, `add`, `mul`, `relu`, `softmax`.

2. **Loss Functions**

   * Implement:

     * Mean Squared Error (for early testing)
     * Cross-Entropy Loss (for language model training)

3. **Optimizers**

   * Start with **SGD**
   * Add **Adam** (you’ll definitely need this for LLMs).

4. **Mini-Batch Support**

   * Implement data batching & shuffling.
