# Simple Neural Network implementation in MIPS using fixed-point arithmetic
# This implements a single-layer neural network (logistic regression)
# for binary classification with 3 input features
# Fixed-point format: Q8.8 (8 bits integer, 8 bits fractional)
# This means we multiply floating point by 256 (2^8)

.data
    # Ensure all data is word-aligned (addresses are multiples of 4)
    # Training data: 4 samples with 3 features each (scaled by 256)
    features:   .word 128, 205, 51      # Sample 1 features (0.5, 0.8, 0.2) * 256
                .word 230, 77, 26       # Sample 2 features (0.9, 0.3, 0.1) * 256
                .word 51, 179, 128      # Sample 3 features (0.2, 0.7, 0.5) * 256
                .word 154, 26, 102      # Sample 4 features (0.6, 0.1, 0.4) * 256
    labels:     .word 256, 256, 0, 0    # Corresponding labels (1.0, 1.0, 0.0, 0.0) * 256
    
    # Model parameters
    weights:    .word 26, 26, 26        # Initial weights (0.1 * 256)
    bias:       .word 0                 # Initial bias (0.0 * 256)
    
    # Hyperparameters
    learning_rate: .word 26             # Learning rate (0.1 * 256)
    epochs:        .word 100            # Number of training epochs
    
    # Output messages
    msg_training: .asciiz "\nTraining neural network...\n"
    msg_complete: .asciiz "Training complete!\n"
    msg_result:   .asciiz "\nPredictions after training:\n"
    msg_sample:   .asciiz "Sample "
    msg_actual:   .asciiz " - Actual: "
    msg_pred:     .asciiz ", Predicted: "
    newline:      .asciiz "\n"
    
    # Temporary storage
    .align 2     # Ensure 4-byte alignment
    prediction:   .word 0
    gradient:     .word 0, 0, 0         # For weight gradients
    bias_grad:    .word 0               # Bias gradient storage
    num_samples:  .word 4
    num_features: .word 3
    fp_one:       .word 256             # 1.0 in fixed point (256)
    fp_zero:      .word 0               # 0.0 in fixed point (0)

.text
.globl main

main:
    # Print training message
    li $v0, 4
    la $a0, msg_training
    syscall
    
    # Load hyperparameters
    lw $t0, epochs      # Number of training epochs
    
    # Start training loop
    li $t1, 0           # epoch counter
    
training_loop:
    # Check if we've completed all epochs
    bge $t1, $t0, training_complete
    
    # Initialize gradients to zero for this epoch
    la $t2, gradient
    li $t3, 0           # Counter for features
    lw $t4, num_features
    
zero_gradients:
    beq $t3, $t4, zero_bias_grad
    sll $t5, $t3, 2     # t5 = t3 * 4 (byte offset)
    add $t6, $t2, $t5   # Address of gradient[t3]
    sw $zero, 0($t6)    # gradient[t3] = 0
    addi $t3, $t3, 1
    j zero_gradients
    
zero_bias_grad:
    la $t2, bias_grad   # Load the address of bias_grad
    sw $zero, 0($t2)    # bias_grad = 0
    
    # Loop through each sample
    li $t2, 0           # sample counter
    lw $t3, num_samples
    
sample_loop:
    beq $t2, $t3, update_params
    
    # Compute the forward pass (prediction)
    jal forward_pass
    
    # Compute the error gradient
    jal compute_gradient
    
    # Next sample
    addi $t2, $t2, 1
    j sample_loop

update_params:
    # Update weights using gradients
    jal update_parameters
    
    # Next epoch
    addi $t1, $t1, 1
    j training_loop

training_complete:
    # Print completion message
    li $v0, 4
    la $a0, msg_complete
    syscall
    
    # Print prediction results
    li $v0, 4
    la $a0, msg_result
    syscall
    
    # Test the model on training data
    li $t2, 0           # sample counter
    lw $t3, num_samples
    
test_loop:
    beq $t2, $t3, exit_program
    
    # Check if index is within bounds
    blt $t2, $t3, continue_test_loop
    
    j exit_program
    
continue_test_loop:
    # Forward pass for this sample
    jal forward_pass
    
    # Print result
    li $v0, 4
    la $a0, msg_sample
    syscall
    
    # Print sample number
    li $v0, 1
    addi $a0, $t2, 1
    syscall
    
    # Print actual label
    li $v0, 4
    la $a0, msg_actual
    syscall
    
    # Load actual label and convert to decimal for display (divide by 256)
    la $t4, labels
    sll $t5, $t2, 2     # t5 = t2 * 4
    add $t4, $t4, $t5   # t4 = address of labels[t2]
    lw $a0, 0($t4)      # a0 = labels[t2]
    
    # Print the integer part
    div $a0, $a0, 256
    li $v0, 1
    syscall
    
    # Print predicted label
    li $v0, 4
    la $a0, msg_pred
    syscall
    
    # Print prediction as integer
    lw $a0, prediction
    div $a0, $a0, 256
    li $v0, 1
    syscall
    
    # Print newline
    li $v0, 4
    la $a0, newline
    syscall
    
    # Next sample
    addi $t2, $t2, 1  # Increment loop counter
    j test_loop

exit_program:
    # Exit program
    li $v0, 10
    syscall



# Forward pass: compute prediction for current sample
forward_pass:
    # Store return address
    addi $sp, $sp, -4
    sw $ra, 0($sp)
    
    # Initialize sum (z) to bias
    lw $t9, bias        # t9 = bias
    
    # Get the base address of current sample features
    la $t4, features
    lw $t8, num_features
    mul $t5, $t2, $t8   # t5 = t2 * 3 (samples * features)
    sll $t5, $t5, 2     # t5 = t5 * 4 (word size)
    add $t4, $t4, $t5   # t4 = address of features[t2][0]
    
    # Get weights address
    la $t6, weights
    
    # Compute weighted sum
    li $t7, 0           # feature counter
    lw $t8, num_features
    
dot_product_loop:
    beq $t7, $t8, apply_threshold
    
    # Load feature and weight
    sll $s1, $t7, 2     # s1 = t7 * 4 (byte offset)
    add $s2, $t4, $s1   # s2 = address of features[t2][t7]
    add $s3, $t6, $s1   # s3 = address of weights[t7]
    
    lw $t0, 0($s2)      # t0 = features[t2][t7]
    lw $t1, 0($s3)      # t1 = weights[t7]
    
    # Multiply and add (with fixed-point adjustment)
    mult $t0, $t1       # t0 = features[t2][t7] * weights[t7]
    mflo $t0            # t0 = LO(result)
    srl $t0, $t0, 8     # t0 = t0 / 256 (adjust fixed point)
    add $t9, $t9, $t0   # t9 += t0 (accumulate)
    
    # Increment counter
    addi $t7, $t7, 1
    j dot_product_loop

apply_threshold:
    # Simple threshold activation function:
    # If z >= 0, prediction = 1.0 (256)
    # If z < 0, prediction = 0.0 (0)
    bgez $t9, set_one
    
    # If z < 0, set prediction to 0
    sw $zero, prediction
    j fp_done
    
set_one:
    # If z >= 0, set prediction to 1.0 (256)
    lw $t0, fp_one
    sw $t0, prediction
    
fp_done:
    # Restore return address and return
    lw $ra, 0($sp)
    addi $sp, $sp, 4
    jr $ra

# Compute gradients for current sample
compute_gradient:
    # Store return address
    addi $sp, $sp, -4
    sw $ra, 0($sp)
    
    # Load prediction and actual label
    lw $t0, prediction
    la $t4, labels
    sll $t5, $t2, 2     # t5 = t2 * 4
    add $t4, $t4, $t5
    lw $t1, 0($t4)      # t1 = labels[t2]
    
    # Calculate error: (prediction - actual)
    sub $t3, $t0, $t1   # t3 = prediction - actual
    
    # Get the base address of current sample features
    la $t4, features
    lw $t8, num_features
    mul $t5, $t2, $t8   # t5 = t2 * 3 (samples * features)
    sll $t5, $t5, 2     # t5 = t5 * 4 (word size)
    add $t4, $t4, $t5   # t4 = address of features[t2][0]
    
    # Accumulate gradients for each weight
    li $t7, 0           # feature counter
    lw $t8, num_features
    
gradient_loop:
    beq $t7, $t8, update_bias_gradient
    
    # Load feature
    sll $s1, $t7, 2     # s1 = t7 * 4 (byte offset)
    add $s2, $t4, $s1   # s2 = address of features[t2][t7]
    lw $t0, 0($s2)      # t0 = features[t2][t7]
    
    # Calculate gradient: error * feature
    mult $t3, $t0       # t3 = error * feature
    mflo $t5            # t5 = LO(result)
    srl $t5, $t5, 8     # t5 = t5 / 256 (adjust fixed-point representation)
    
    # Accumulate to gradient array
    la $t9, gradient
    sll $s3, $t7, 2     # s3 = t7 * 4
    add $t9, $t9, $s3   # t9 = address of gradient[t7]
    lw $t6, 0($t9)      # t6 = current gradient value
    addu $t6, $t6, $t5   # t6 += t5
    sw $t6, 0($t9)      # gradient[t7] = t6
    
    # Next feature
    addi $t7, $t7, 1
    j gradient_loop

update_bias_gradient:
    # Save the value of t3 (error) to be safe
    move $s4, $t3       # Temporarily store error in s4
    
    # Update bias gradient (just add the error)
    la $t9, bias_grad   # Get address of bias_grad
    lw $t6, 0($t9)      # t6 = bias_grad value
    addu $t6, $t6, $t3   # t6 += error
    sw $t6, 0($t9)      # Store updated value back to bias_grad
    
    # Restore return address and return
    lw $ra, 0($sp)
    addi $sp, $sp, 4
    jr $ra


# Update weights and bias using gradients
update_parameters:
    # Store return address
    addi $sp, $sp, -4
    sw $ra, 0($sp)
    
    # Load learning rate
    lw $s0, learning_rate
    
    # Update weights
    li $t3, 0           # feature counter
    lw $t4, num_features
    
update_weights_loop:
    beq $t3, $t4, update_bias
    
    # Get current weight
    la $t5, weights
    sll $t6, $t3, 2     # t6 = t3 * 4
    add $t5, $t5, $t6   # t5 = address of weights[t3]
    lw $t1, 0($t5)      # t1 = weights[t3]
    
    # Get gradient
    la $t6, gradient
    sll $t7, $t3, 2     # t7 = t3 * 4
    add $t6, $t6, $t7   # t6 = address of gradient[t3]
    lw $t2, 0($t6)      # t2 = gradient[t3]
    
    # Multiply gradient by learning rate
    mult $t2, $s0       # t2 = gradient[t3] * learning_rate
    mflo $t2            # t2 = LO(result)
    srl $t2, $t2, 8     # t2 = t2 / 256 (adjust fixed-point representation)
    
    # Update weight: weight = weight - learning_rate * gradient
    sub $t1, $t1, $t2   # t1 = weights[t3] - (gradient[t3] * learning_rate)
    sw $t1, 0($t5)      # weights[t3] = t1
    
    # Next weight
    addi $t3, $t3, 1
    j update_weights_loop

update_bias:
    # Update bias: bias = bias - learning_rate * bias_gradient
    lw $t1, bias        # t1 = bias
    la $t0, bias_grad   # Get address of bias_grad
    lw $t2, 0($t0)      # t2 = bias_gradient value
    mult $t2, $s0       # t2 = bias_gradient * learning_rate
    mflo $t2            # t2 = LO(result)
    srl $t2, $t2, 8     # t2 = t2 / 256 (adjust fixed-point representation)
    sub $t1, $t1, $t2   # t1 = bias - (bias_gradient * learning_rate)
    sw $t1, bias        # bias = t1
    
    # Restore return address and return
    lw $ra, 0($sp)
    addi $sp, $sp, 4
    jr $ra
