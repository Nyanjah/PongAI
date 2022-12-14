use bevy::sprite::collide_aabb::collide;// collision detection between rects
use bevy::{prelude::*, sprite::collide_aabb::Collision}; // Bevy engine
pub mod network;   // custom made module for the network instantiation and calculations
pub mod reinforce; // custom made module for implementing the REINFORCE algorithm
use rand::Rng;     // for randomly generated values

use reinforce::*; 
use network::*;
use super::*;
use rand::seq::SliceRandom;
// Bevy is an ECS (Entinity-Component-System) data-driven rust game engine
// the functions defined in this file are "systems" that act on the "components"
// of an "entity". An entity in this context is just a bundle of components with an ID.
// The arguments in these functions are queries for existing entities in the game, 
// such as the paddle, ball, score, etc. The functions then iterate over the results of the query
// and mutate their components, such as the paddles position, ball's velocity, and score's value.

// These functions are either run at the start of the app's execution or every frame. In this context,
// the network must query for entitys which encode the game's state and use the components of those entities
// to form the networks input activations. All of the systems corresponding to the network can be found in the 
// module network.rs, most everything in this file handles the main game logic.


#[derive(Default, Resource)]
pub struct Score {
    pub pc: u32,
    pub npc: u32,
    pub pc_wins: u32,
    pub npc_wins: u32,
    pub epoch: u64,
}

#[derive(Component)]
pub struct Visual;

#[derive(Component, PartialEq, Eq, Debug)]
pub enum Paddle {
    PC,
    NPC,
}

#[derive(Component)]
pub struct Velocity {
    pub(crate) x: f32,
    pub(crate) y: f32,
}


pub fn spawn_camera(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}

pub fn spawn_visuals(mut commands: Commands) {
    for i in [-1.0, 1.0] {
        commands.spawn((
            SpriteBundle {
                sprite: Sprite {
                    color: Color::rgb(1.0, 0.0, 0.0),
                    custom_size: Some(Vec2::new(10.0, HEIGHT)),
                    ..default()
                },
                transform: Transform::from_translation(Vec3::new((WIDTH / 2.0) * i, 0.0, 0.0)),
                ..default()
            },
            Visual,
        ));
    }
}

pub fn spawn_paddles(mut commands: Commands) {
    for i in [-1.0, 1.0] {
        commands.spawn((
            SpriteBundle {
                sprite: Sprite {
                    color: Color::rgb(1.0, 1.0, 1.0),
                    custom_size: Some(PADDLE_SIZE),
                    ..default()
                },
                transform: Transform::from_translation(Vec3::new((WIDTH / 2.0) * i, 0.0, 1.0)),
                ..default()
            },
            {
                if i == 1.0 {
                    Paddle::NPC
                } else {
                    Paddle::PC
                }
            },
        ));
    }
}

pub fn spawn_ball(mut commands: Commands) {
    let mut rng = rand::thread_rng();
    let random_x =  [-1.0, 1.0].choose(&mut rng).unwrap()* BALL_SPEED;
    let random_y = [-1.0, 1.0].choose(&mut rng).unwrap() * BALL_SPEED;
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::rgb(1.0, 1.0, 1.0),
                custom_size: Some(BALL_SIZE),
                ..default()
            },
            ..default()
        },
        Velocity {
            x: random_x,
            y: random_y,
        },
    ));
}

pub fn spawn_text(mut commands: Commands, asset_server: Res<AssetServer>) {
    let text_style = TextStyle {
        font: asset_server.load("LeagueSpartan-Bold.otf"),
        font_size: 30.0,
        color: Color::WHITE,
    };
    commands.spawn(Text2dBundle {
        text: Text::from_section("0        0", text_style.clone())
            .with_alignment(TextAlignment::CENTER),
        ..default()
    });
}

pub fn move_paddles(mut paddles: Query<(&mut Transform, &Paddle)>, 
    keys: Res<Input<KeyCode>>,
    network_keypress: Res<NPCInput>
    ) {
    for mut trans in paddles.iter_mut() {
        if *trans.1 == Paddle::PC {
            if keys.pressed(KeyCode::W) {
                if trans.0.translation.y <= (HEIGHT / 2.0 - PADDLE_SIZE[1] / 2.0) {
                    trans.0.translation.y += PADDLE_SPEED;
                }
            }
            if keys.pressed(KeyCode::S) {
                if trans.0.translation.y >= (-HEIGHT / 2.0 + PADDLE_SIZE[1] / 2.0) {
                    trans.0.translation.y -= PADDLE_SPEED;
                }
            }
        }
        if *trans.1 == Paddle::NPC {
            if *network_keypress == NPCInput::UpKey {
                if trans.0.translation.y <= (HEIGHT / 2.0 - PADDLE_SIZE[1] / 2.0) {
                    trans.0.translation.y += PADDLE_SPEED;
                }
            }
            if *network_keypress == NPCInput::DownKey  {
                if trans.0.translation.y >= (-HEIGHT / 2.0 + PADDLE_SIZE[1] / 2.0) {
                    trans.0.translation.y -= PADDLE_SPEED;
                }
            }
        }
    }
}

pub fn move_ball(
    mut query: Query<(&mut Transform, &mut Velocity)>,
    mut score: ResMut<Score>,
    mut epoch_data: ResMut<EpochData>
) {
    for (mut trans, mut velocity) in query.iter_mut() {
        if trans.translation.x.abs() >= WIDTH / 2.0 - BALL_SIZE[0] / 2.0 {
            // If it went to the left, npc earned a point
            if trans.translation.x < 0.0 {
                score.npc = score.npc + 1;
            }
            // If it went to the right pc earned a point
            if trans.translation.x > 0.0 {
                score.pc = score.pc + 1;
                // // The network failed to hit the ball, so punish it.
                // epoch_data.rewards.pop();
                // epoch_data.rewards.push(-1.0);
                // println!("Network Punished -1.0")
            }
            trans.translation.x = 0.0;
            trans.translation.y = 0.0;
        }
        if trans.translation.y.abs() >= HEIGHT / 2.0 - BALL_SIZE[0] / 2.0 {
            velocity.y = velocity.y * -1.00;
        }

        trans.translation.x += velocity.x;
        trans.translation.y += velocity.y;
    }

}

pub fn handle_collisions(
    mut balls: Query<(&mut Transform, &mut Velocity, Without<Paddle>)>,
    paddles: Query<(&mut Transform,&Paddle, Without<Velocity>, With<Paddle>)>,
    mut epoch_data: ResMut<EpochData>
) {
    for mut ball in balls.iter_mut() {
        // Note to self: ONLY HAVE THIS SECTION UNCOMMENTED FOR TRAINING THE NETWORK!
        if TRAINING{
        // If the ball would get past the player-paddle:
            if ball.0.translation.x < -1.0*(( WIDTH / 2.0 - BALL_SIZE[0] / 2.0)) {
                // Hit the ball anyway to simulate an opponent to train against
                let mut rng = rand::thread_rng();
                    if rng.gen_range(0.0..1.00) <= TRAINER_HIT_RATE{
                    ball.1.x = ball.1.x * -1.00;
                    ball.0.translation.x = ball.0.translation.x + ball.1.x;
                    ball.0.translation.y = ball.0.translation.y + ball.1.x;
                }
            }
        }
        for paddle in paddles.iter() {
            let collision_detection = collide(
                ball.0.translation,
                BALL_SIZE,
                paddle.0.translation,
                PADDLE_SIZE,
            );
            if collision_detection == Some(Collision::Right)
                || collision_detection == Some(Collision::Left)
            {
                ball.1.x = ball.1.x * -1.00;
                ball.0.translation.x = ball.0.translation.x + ball.1.x;
                ball.0.translation.y = ball.0.translation.y + ball.1.x;

            } else if collision_detection == Some(Collision::Top)
                || collision_detection == Some(Collision::Bottom)
            {
                ball.1.x = ball.1.x * -1.00;
                ball.1.y = ball.1.y * -1.00;
                ball.0.translation.x = ball.0.translation.x + ball.1.x;
                ball.0.translation.y = ball.0.translation.y + ball.1.x;
            }
        }

    }
}

pub fn update_score_text(
    mut text_query: Query<&mut Text>,
    mut score: ResMut<Score>,
    mut epoch_data : ResMut<EpochData>
) {
    for mut text in text_query.iter_mut() {
        // If the AI WON
        if score.npc > 10 {
            score.pc = 0;
            score.npc = 0;
            score.npc_wins = score.npc_wins + 1;
            epoch_data.epoch = epoch_data.epoch + 1;
            if TRAINING{
                // As we finished a game, the epoch is over for now.
                epoch_data.inprogress = false;
                // Reward the network
                println!("Network Won, Reward +1.0");
                epoch_data.rewards.push(1.0);
                text.sections[0].value = format!("{}        {}", score.pc, score.npc);
            }
        }
        // If the AI LOST
        else if  score.pc > 10 {
            score.pc = 0;
            score.npc = 0;
            score.pc_wins = score.pc_wins + 1;
            epoch_data.epoch = epoch_data.epoch + 1;
            if TRAINING{
                // As we finished a game, the epoch is over for now.
                epoch_data.inprogress = false;
                // Punish the network
                println!("Network Lost, Reward -1.0");
                epoch_data.rewards.push(-1.0);
                text.sections[0].value = format!("{}        {}", score.pc, score.npc); 
            } 
        }
        // 
        else{
            // The network gets no reward or punishment
            if TRAINING{
                epoch_data.rewards.push(0.0);
            }
            text.sections[0].value = format!("{}        {}", score.pc, score.npc);
        }
        
    }
}

