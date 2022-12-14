mod pong;
use pong::*;
use pong::network::*;
use pong::reinforce::*;
use bevy::prelude::*;

// Training mode ( training the network or playing against it)
pub const TRAINING:bool = true;

// This determines the probability of the player hitting the ball back when playing
// against the network in training mode. This is intended to simulate an opponent of 
// a configurable level of skill.
pub const TRAINER_HIT_RATE:f32 = 0.5;

// Network training parameters
pub const DISCOUNTING_FACTOR: f32 = 0.99;
pub const LEARNING_RATE: f32 = 0.15;

// Game-defining constants
pub const WIDTH: f32 = 720.0;
pub const HEIGHT: f32 = 480.0;
pub const PADDLE_SIZE: Vec2 = Vec2::new(0.027 * WIDTH, 0.185 * HEIGHT);
pub const BALL_SIZE: Vec2 = Vec2::new(PADDLE_SIZE.x / 2.0, PADDLE_SIZE.x / 2.0);    
pub const BALL_SPEED: f32 = 4.00;
pub const PADDLE_SPEED: f32 = 4.0 * HEIGHT / WIDTH;


fn main() {
    let present_mode = ||{if (TRAINING){bevy::window::PresentMode::Immediate} else {bevy::window::PresentMode::Fifo}};
    App::new()
        .insert_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        .insert_resource(Score { pc: 0, npc: 0 , pc_wins: 0 ,npc_wins: 0, epoch: 0})
        .insert_resource(EpochData{
            actions: Vec::new(),
            states: Vec::new(),
            rewards:Vec::new(),
            inprogress: TRAINING,
            epoch:0
        })
        .insert_resource(PolicyGradient::default())
        .insert_resource(NPCInput::UpKey)
        .add_startup_system(spawn_camera)
        .add_startup_system(spawn_paddles)
        .add_startup_system(spawn_ball)
        .add_startup_system(spawn_visuals)
        .add_startup_system(spawn_text)
        .add_startup_system(initialize_network)
        //.add_startup_system(spawn_chart)
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            window: WindowDescriptor {
                height: HEIGHT,
                width: WIDTH,
                title: "EE456 Final Project - Dev Branch".to_string(),
                resizable: false,
                // Note: Present mode impacts game speed so immediate should be used for training.
                present_mode: (present_mode()),
                ..default()
            },
            ..default()
        }))
        .add_system(move_paddles)
        .add_system(handle_collisions.before(move_ball).before(move_paddles))
        .add_system(move_ball.after(handle_collisions))
        .add_system(update_score_text)
        .add_system(feed_forward.before(move_paddles))
        .add_system(output_network_state)
        .add_stage_after(CoreStage::Update, "Util", SystemStage::single_threaded())
        .add_system_to_stage("Util",train_with_reinforce)

        .run();
}

// Note:

// Collecting information about the game's state and the policy output is done in two decoupled systems:
// feed_forward which runs in the "Core" stage, and train_with_reinforce which runs in the "Util" stage.
// Changing when these systems run per frame may cause dependency issues when tracking the trajectory
// which could lead to a disconnect between state and action. These should not be modified.