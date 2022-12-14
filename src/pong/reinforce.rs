use super::*;

// This file contains my implementation of the REINFORCE algorithm.

// The policy (network) maps a state s to an action a. ( p: s -> a )
// An epoch entails generating a "trajectory" of states, actions, and their corresponding rewards:
//   S_0, A_0, R_1, ... , S_(T-1), A_(T-1), R_T, following the policy p. (in this case the network).


// The algorithm loops over each time step of the epoch ( t = 0, 1, 2, ..., T - 1):
// and computes:
// - The Discounted Return -
// * This is the sum of all the rewards from [ t + 1 , T] multiplied by powers of the discount factor.
// This is a solution to the "credit assignment problem". Basically, its the problem of deciding which
// actions to reward to get the network to perform better. In this case, I want the actions to be rewarded
// more heavily as the ball is approaching the paddle. So when a reward is given for hitting the ball the
// further back in the time actions were performed the more their reward values will be "discounted".

// - The Policy Gradient -
// * This determines how my weights and biases will change to maximize the rewards.
// It is calculated using the gradient of the logorithmic function of the policy's output,
// scaled by the Discounted Return Value calculated in the previous step.
// This also gets scaled by the learning rate before being added to the network's parameters.

// Note:  I was confused by this at first, but the policy output that is used in its calculation
// is not the action itself, but rather the probability of the action which was selected.

// The idea of the algorithm is that it estimates the expected return for a generated trajectory
// and uses that to estimate the gradient of the expected return with respect to the
// parameters of the network.

#[derive(Resource)]
pub struct EpochData {
    // Vector of tuples containing the decided action and its probability of being sampled
    pub actions: Vec<(NPCInput, f32)>,
    // Vector of states which contain the inputs into the network
    pub states: Vec<[f32; 5]>,
    // Vector of rewards which containg the reward assigned to each action
    pub rewards: Vec<f32>,
    // Boolean to check if an epoch is still in progress
    pub inprogress: bool,
    // Integer to track the number of epochs which have passed
    pub epoch: u32,
}

#[derive(Resource, Default)]
pub struct PolicyGradient {
    pub biases: [f32; 11],
    pub first_layer_weights: [[f32; 5]; 5],
    pub second_layer_weights: [[f32; 5]; 5],
    pub output_layer_weights: [f32; 5],
}

pub fn train_with_reinforce(
    mut network: ResMut<Network>,
    mut gradient: ResMut<PolicyGradient>,
    mut epochdata: ResMut<EpochData>,
) {
    // If an epoch just ended, apply the algorithm
    // The policy has generated a trajectory which is stored in the state, action, and reward vectors of EpochData.
    if (epochdata.inprogress == false) && TRAINING {
        let mut discounted_returns: Vec<f32> = Vec::new();
        // For each time step of the epoch
        for i in 0..epochdata.states.len() {
            // Creating a new entry in our discounted returns vector
            discounted_returns.push(0.0);
            // Sum up the changes to that entry
            for j in (i)..epochdata.states.len() {
                // Applying the discount factor to the rewards distributed over time
                // (Estimating an expected return using the trajectory)
                discounted_returns[i] = discounted_returns[i] + epochdata.rewards[j] * (DISCOUNTING_FACTOR.powf((j - i) as f32));
            }
            // Calculating the loss / error for the action taken by the network at the current timestep
            // Note: Since gradient descent minimizes loss, this gets multiplied by -1.00 because we want to maximize this product
            // to increase the probability of the network outputting a sequence of actions which yield greater rewards

            // Calculate the policy gradient to reinforce the sequence of actions that led to the rewards
            // First we start with the forward pass:
            let network_inputs = epochdata.states[i];
            // Arrays to store the activations for the 1st and 2nd layers during the forward pass
            let mut activations: [[f32; 5]; 2] = [[0.0; 5]; 2];
            // For each input value
            for i in 0..5 {
                // For each node in the first hidden layer
                for k in 0..5 {
                    activations[0][k] =
                        activations[0][k] + network_inputs[i] * network.first_layer_weights[i][k];
                }
            }
            // Applying bias and activation function to the outputs
            for i in 0..5 {
                activations[0][i] = activation(activations[0][i] + network.biases[i]);
            }
            // For each output from the first layer
            for i in 0..5 {
                // For each node in the second hidden layer
                for k in 0..5 {
                    activations[1][k] =
                        activations[1][k] + activations[0][i] * network.second_layer_weights[i][k];
                }
            }
            // Applying bias and activation function to the outputs
            for i in 0..5 {
                activations[1][i] = activation(activations[1][i] + network.biases[i + 5]);
            }
            let mut pi: f32 = 0.0;
            // For each output from the second layer
            for i in 0..5 {
                pi = pi + activations[1][i] * network.output_layer_weights[i];
            }
            // Applying the output node's bias and activation function to the final output value
            pi = activation(pi + network.biases[10]);

            //----------------------------------------------------------------------------------------------------
            // Now that I've computed the forward pass, I can move on to the backwards pass
            // Since we want the probability of the action which was taken, if the action taken was
            // down we actually want the policy output pi to be 1.0 - prob of going up.
            let mut adjustment =
                LEARNING_RATE * (DISCOUNTING_FACTOR.powf(i as f32)) * discounted_returns[i];

            if epochdata.actions[i].0 == NPCInput::DownKey {
                pi = (1.0 - pi);
                // We also want the negative of the gradient if the action was down, since increasing the chances
                // of the down action corresponds to decreasing the chances of the up action.
                adjustment = adjustment * -1.00;
            }

            // Calculating update for the bias of the output node
            gradient.biases[10] = gradient.biases[10] + adjustment * (1.0 - pi);

            for k in 0..5 {
                // Calculating update for weights in the output layer
                gradient.output_layer_weights[k] =
                    gradient.output_layer_weights[k] + adjustment * (1.0 - pi) * activations[1][k];
                // Calculating updates for the biases of the second hidden layer
                gradient.biases[k + 5] = gradient.biases[k + 5]
                    + adjustment * (1.0 - pi) * activations[1][k] * (1.0 - activations[1][k]);
            }

            for j in 0..5 {
                // Calculating updates for the weights between the two hidden layers
                for k in 0..5 {
                    gradient.second_layer_weights[j][k] = gradient.second_layer_weights[j][k]
                        + adjustment
                            * (1.0 - pi)
                            * network.output_layer_weights[k]
                            * activations[1][k]
                            * (1.0 - activations[1][k])
                            * activations[0][j]
                }
                // Calculating updates for the biases in the first hidden layer ( this is where things get complicated!)
                let mut sum_results: f32 = 0.0;
                for k in 0..5 {
                    sum_results = sum_results
                        + network.output_layer_weights[k]
                            * activations[1][k]
                            * network.second_layer_weights[j][k]
                            * (1.0 - activations[1][k]);
                }
                gradient.biases[j] = gradient.biases[j]
                    + adjustment
                        * (1.0 - pi)
                        * activations[0][j]
                        * (1.0 - activations[0][j])
                        * sum_results;
            }
            // This is were things get really rough- but the gradient calculations can't lie so I'll do what they say...
            // Calculating the updates to the weights between the input and first hidden layer
            for m in 0..5 {
                for j in 0..5 {
                    let mut sum_results: f32 = 0.0;
                    for k in 0..5 {
                        sum_results = sum_results
                            + network.output_layer_weights[k]
                                * activations[1][k]
                                * network.second_layer_weights[j][k]
                                * (1.0 - activations[1][k]);
                    }
                    gradient.first_layer_weights[m][j] = gradient.first_layer_weights[m][j] + adjustment
                        * (1.0 - pi)
                        * network_inputs[m]
                        * activations[0][j]
                        * (1.0 - activations[0][j])
                        * sum_results;
                }
            }
        }
        // Now that I have computed the sum of the gradiens for each time step in the last epoch's trajectory,
        // I can apply them to the network:
        for j in 0..5 {
            for k in 0..5 {
                // Updating weights
                network.first_layer_weights[j][k] =
                    network.first_layer_weights[j][k] + gradient.first_layer_weights[j][k];
                network.second_layer_weights[j][k] =
                    network.second_layer_weights[j][k] + gradient.second_layer_weights[j][k];
            }
            network.output_layer_weights[j] =
                network.output_layer_weights[j] + gradient.output_layer_weights[j];
        }
        for j in 0..11 {
            // Updating baises
            network.biases[j] = network.biases[j] + gradient.biases[j]
        }
        // Setting the gradient struct back to zero
        *gradient = PolicyGradient::default();

        // Now that we have updated the network, we can continue to the next epoch.
        // Setting the global epoch-in-progress flag back to true:
        epochdata.inprogress = true;
        // Flushing the trajectory data stored in the action, reward, and state buffers:
        epochdata.actions = Vec::new();
        epochdata.rewards = Vec::new();
        epochdata.states = Vec::new();
        // Debug output:
        //println!("Discounted returns: {:?}", discounted_returns)
    }
}
