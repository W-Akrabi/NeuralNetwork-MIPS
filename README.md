# NeuralNetwork-MIPS
# 🧠 Neural Network in MIPS (Using Fixed-Point Math)

This is a simple neural network (more like a logistic regression) written entirely in **MIPS Assembly**. It’s designed to classify binary labels based on 3 input features per sample. Since MIPS doesn’t support floating-point operations by default, everything here is done using **fixed-point arithmetic (Q8.8 format)**.

---

## 🔍 What This Project Does

- Takes in 4 training samples, each with 3 features and a label (0 or 1)
- Runs a forward pass, calculates prediction errors
- Updates weights and bias over multiple epochs
- Uses a super basic threshold function to turn predictions into 0s and 1s
- Prints predictions alongside actual labels after training

---

## 🧮 Fixed-Point Format (Q8.8)

Since there's no floating-point math in basic MIPS, we’re using fixed-point instead:
- Multiply your float by 256 before storing (e.g. 0.5 becomes 128)
- Think of 256 as representing 1.0
- When printing results, divide by 256 to get the "real" number

---

## 🧪 Training Data (Already Converted to Fixed-Point)

| Sample | Feature 1 | Feature 2 | Feature 3 | Label |
|--------|-----------|-----------|-----------|-------|
| 1      | 0.5       | 0.8       | 0.2       | 1.0   |
| 2      | 0.9       | 0.3       | 0.1       | 1.0   |
| 3      | 0.2       | 0.7       | 0.5       | 0.0   |
| 4      | 0.6       | 0.1       | 0.4       | 0.0   |

---

## ⚙️ How It Works (Step-by-Step)

1. **Forward pass**: Multiply each input feature by its corresponding weight, add everything up with the bias.
2. **Apply threshold**: If the result is >= 0.5, output is 1. Otherwise, 0.
3. **Backprop (kinda)**: Subtract actual label from prediction to get error.
4. **Update weights**: Use the error to adjust weights and bias.
5. Repeat this for all samples and for 100 epochs.

---

## ▶️ How to Run It

You’ll need a MIPS simulator like:
- [MARS](http://courses.missouristate.edu/kenvollmar/mars/)
- [QtSPIM](https://sourceforge.net/projects/spimsimulator/)
- [Saturn](https://github.com/1whatleytay/saturn)

**Steps:**
1. Open the `.asm` file in the simulator
2. Assemble the file
3. Run it
4. Check the output console for predictions and labels
