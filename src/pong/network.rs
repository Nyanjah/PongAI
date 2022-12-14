use super::*;
use rand::{Rng};
use serde::{Serialize, Deserialize};


// This file contains the neural network model trained with a reward policy implementing backpropogation.
// It takes in the following values as inputs:
// - The y position of the NPC paddle
// - The <x,y> co-ordinates of the ball
// - The <x,y> velocity of the ball
// It then chooses between two actions (UP/DOWN) to control the game paddle and receive rewards
// which adjust the weights and baises over multiple rounds.
//-----------------------------------------------------------------------------------------------------
// Diagram:
//  *Inputs get fed into network*
// x_pos of ball ->                * Network outputs a
// y_pos of ball ->               Probability of Actions*       *Action Sampled*     *Game reacts to action*
// x_vel of ball ->>>>  [NETWORK]-- > [0.0 , 1.0 ] ----->   Res<Some(enum:NPC_Input)> --> Game State
// y_vel of ball ->         ^                                                                 |
// y_pos of paddle ->       |                       *Feedback Loop*                           |
//                          |                   state, reward for action                      |
//                          |_________________________________________________________________|
//
//            *weights get updated every epoch using calculated adjustments via REINFORCE algorithm ~21 rounds*
//                        

pub fn activation(input: f32) -> f32 {
    // activation returns sigmoid of given value
    return 1.0 / (1.0 + input.exp());
}

#[derive(Resource,Serialize, Deserialize, Debug)]
pub struct Network {
    pub biases: [f32; 11],
    pub first_layer_weights: [[f32; 5]; 5],
    pub second_layer_weights: [[f32; 5]; 5],
    pub output_layer_weights: [f32; 5],
}


#[derive(Resource, PartialEq, Eq, Copy, Clone, Debug)]
pub enum NPCInput {
    UpKey,
    DownKey,
}

// System to initialize the network's state
pub fn initialize_network(mut commands: Commands) {
    // Initializes the network's weights and biases to random values
    let mut rng = rand::thread_rng();
    // Setting the range to select the initial values from
    let genrange = -0.05..0.05;
    let mut first_layer_weights: [[f32; 5]; 5] = [[0.0; 5]; 5];
    let mut second_layer_weights: [[f32; 5]; 5] = [[0.0; 5]; 5];
    let mut output_layer_weights: [f32; 5] = [0.0; 5];
    let mut biases: [f32; 11] = [0.0; 11];

    for i in 0..5 {
        for k in 0..5 {
            first_layer_weights[i][k] = rng.gen_range(genrange.clone());
        }
    }
    for i in 0..5 {
        for k in 0..5 {
            second_layer_weights[i][k] = rng.gen_range(genrange.clone());
        }
    }
    for i in 0..5 {
        output_layer_weights[i] = rng.gen_range(genrange.clone());
    }
    for i in 0..11 {
        biases[i] = rng.gen_range(genrange.clone());
        
    }

    commands.insert_resource(Network{
        biases: biases,
        first_layer_weights: first_layer_weights,
        second_layer_weights: second_layer_weights,
        output_layer_weights: output_layer_weights,
    });

    commands.insert_resource(PolicyGradient{
        biases: [0.0;11],
        first_layer_weights: [[0.0;5];5],
        second_layer_weights: [[0.0;5];5],
        output_layer_weights: [0.0;5],
    })

}

pub fn spawn_chart(
    mut commands: Commands
){
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::rgb(0.0, 1.0, 0.0),
                custom_size: Some(Vec2::new(PADDLE_SIZE.x,10.0)),
                ..default()
            },
            ..default()
        },
    ));
}


// System to calculate the networks output given the game's state between frames
// and sample the final action to be used in the next frame
pub fn feed_forward(
    mut action: ResMut<NPCInput>,
    network: ResMut<Network>,
    paddle_query: Query<(&mut Transform, &Paddle, With<Paddle>, Without<Velocity>)>,
    ball_query: Query<(&mut Transform, &mut Velocity, With<Velocity>, Without<Paddle>)>,
    mut epoch_data: ResMut<EpochData>
) {
    // Extracting the networks input values from the components of the gamestate
    let mut network_inputs: [f32; 5] = [0.0; 5];

    for paddle in paddle_query.iter() {
        if *paddle.1 == Paddle::NPC {
            // NPC Paddle y-value
            network_inputs[4] = paddle.0.translation.y;
        }
    }
    for ball in ball_query.iter() {
        network_inputs[0] = ball.0.translation.x; // x-pos
        network_inputs[1] = ball.0.translation.y; // y-pos
        network_inputs[2] = ball.1.x;             // x-vel
        network_inputs[3] = ball.1.y;             // y-vel
    }
    // Updating the state in epoch_data
    epoch_data.states.push(network_inputs);
    // Array to hold the computed values used for calculating the networks output
    let mut first_layer_outputs: [f32; 5] = [0.0; 5];
    
    // For each input value
    for i in 0..5{
        // For each node in the first hidden layer
        for k in 0..5{
            first_layer_outputs[k] = first_layer_outputs[k] + network_inputs[i] * network.first_layer_weights[i][k];
        }
    }
    // Applying bias and activation function to the outputs
    for i in 0..5{
        first_layer_outputs[i] = activation(first_layer_outputs[i] + network.biases[i]);
    }
    let mut second_layer_outputs: [f32; 5] = [0.0; 5];

    // For each output from the first layer
    for i in 0..5{
        // For each node in the second hidden layer
        for k in 0..5{
            second_layer_outputs[k] = second_layer_outputs[k] + first_layer_outputs[i] * network.second_layer_weights[i][k];
        }
    }
    // Applying bias and activation function to the outputs
    for i in 0..5{
        second_layer_outputs[i] = activation(second_layer_outputs[i] + network.biases[i+5]);
    }
    let mut network_output_value:f32 = 0.0;
    // For each output from the second layer
    for i in 0..5{
        network_output_value = network_output_value + second_layer_outputs[i] * network.output_layer_weights[i];
    }
    // Applying the output node's bias and activation function to the final output value
    network_output_value = activation(network_output_value + network.biases[10]);

    // Sampling from the output and updating the action to be taken by the network...
    let mut rng = rand::thread_rng();  
    let random_value = rng.gen_range(0.0..1.0);

    if random_value < network_output_value{
        *action = NPCInput::UpKey;
        //println!("Output: {},  sampled {} ---> Decided: UP",network_output_value,random_value);
        epoch_data.actions.push((NPCInput::UpKey,network_output_value)) // Update epoch data with the chosen action
    }
    else{
        *action = NPCInput::DownKey;
        //println!("Output: {},  sampled {} ---> Decided: DOWN",network_output_value,random_value);
        epoch_data.actions.push((NPCInput::DownKey,1.0 - network_output_value)) // Update epoch data with the chosen action
    }
}

pub fn output_network_state(
    keys: Res<Input<KeyCode>>,
    network: Res<Network>,
){
    if keys.pressed(KeyCode::X){
        let serialized = serde_json::to_string(&(*network)).unwrap();
        println!("{}", serialized);
    }

}

