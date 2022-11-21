use bevy::sprite::collide_aabb::collide;
use bevy::{prelude::*, sprite::collide_aabb::Collision};
// Bevy is an ECS (Entinity-Component-System) data-driven rust game engine
use rand::seq::SliceRandom;
mod network;

pub const WIDTH: f32 = 1080.0;
pub const HEIGHT: f32 = 720.0;
pub const PADDLE_SIZE: Vec2 = Vec2::new(0.027 * WIDTH, 0.185 * HEIGHT);
pub const BALL_SIZE: Vec2 = Vec2::new(PADDLE_SIZE.x / 2.0, PADDLE_SIZE.x / 2.0);
pub const BALL_SPEED: f32 = 4.00;
pub const PADDLE_SPEED: f32 = 4.0 * HEIGHT / WIDTH;

#[derive(Component)]
struct Visual;

#[derive(Component)]
struct Ball;

#[derive(Component, PartialEq, Eq)]
enum Paddle {
    PC,
    NPC,
}

#[derive(Component)]
struct Velocity {
    x: f32,
    y: f32,
}

#[derive(Default, Resource)]
struct Score {
    pc: i32,
    npc: i32,
}

#[derive(Resource)]
enum NPC_Input{
    UP_KEY,
    DOWN_KEY,
}



fn spawn_camera(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}

fn spawn_visuals(mut commands: Commands) {
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

fn spawn_paddles(mut commands: Commands) {
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
                    Paddle::PC
                } else {
                    Paddle::NPC
                }
            },
        ));
    }
}

fn spawn_ball(mut commands: Commands) {
    let mut rng = rand::thread_rng();
    let random_x = [-1.0, 1.0].choose(&mut rng).unwrap() * BALL_SPEED;
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

fn spawn_text(mut commands: Commands, asset_server: Res<AssetServer>) {
    let text_style = TextStyle {
        font: asset_server.load("LeagueSpartan-Bold.otf"),
        font_size: 100.0,
        color: Color::WHITE,
    };
    commands.spawn(Text2dBundle {
        text: Text::from_section("0        0", text_style.clone())
            .with_alignment(TextAlignment::CENTER),
        ..default()
    });
}

fn move_paddles(mut paddles: Query<(&mut Transform, &Paddle)>, keys: Res<Input<KeyCode>>) {
    for mut trans in paddles.iter_mut() {
        if *trans.1 == Paddle::NPC {
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
        if *trans.1 == Paddle::PC {
            if keys.pressed(KeyCode::Up) {
                if trans.0.translation.y <= (HEIGHT / 2.0 - PADDLE_SIZE[1] / 2.0) {
                    trans.0.translation.y += PADDLE_SPEED;
                }
            }
            if keys.pressed(KeyCode::Down) {
                if trans.0.translation.y >= (-HEIGHT / 2.0 + PADDLE_SIZE[1] / 2.0) {
                    trans.0.translation.y -= PADDLE_SPEED;
                }
            }
        }
    }
}

fn move_ball(mut query: Query<(&mut Transform, &mut Velocity)>, mut score: ResMut<Score>) {
    for (mut trans, mut velocity) in query.iter_mut() {
        if trans.translation.x.abs() >= WIDTH / 2.0 - BALL_SIZE[0] / 2.0 {
            // If it went to the left, npc won
            if trans.translation.x < 0.0 {
                score.pc = score.pc + 1;
            }
            // If it went to the right pc won
            if trans.translation.x > 0.0 {
                score.npc = score.npc + 1;
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

fn handle_collisions(
    mut balls: Query<(&mut Transform, &mut Velocity, Without<Paddle>)>,
    paddles: Query<&mut Transform, With<Paddle>>,
) {
    for mut ball in balls.iter_mut() {
        for paddle in paddles.iter() {
            let collision_detection = collide(
                ball.0.translation,
                BALL_SIZE,
                paddle.translation,
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

fn update_score_text(mut text_query: Query<&mut Text>, score: Res<Score>) {
    for mut text in text_query.iter_mut() {
        text.sections[0].value = format!("{}        {}", score.npc, score.pc);
    }
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        .insert_resource(Score { pc: 0, npc: 0 })
        .insert_resource(NPC_Input::UP_KEY)
        .add_startup_system(spawn_camera)
        .add_startup_system(spawn_paddles)
        .add_startup_system(spawn_ball)
        .add_startup_system(spawn_visuals)
        .add_startup_system(spawn_text)
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            window: WindowDescriptor {
                height: HEIGHT,
                width: WIDTH,
                title: "EE456 Final Project".to_string(),
                resizable: false,
                // Note to self: Uncomment this to disable Vsync for faster training times...
                //present_mode: bevy::window::PresentMode::Immediate,
                ..default()
            },
            ..default()
        }))
        .add_system(move_paddles)
        .add_system(handle_collisions)
        .add_system(move_ball)
        .add_system(update_score_text)
        .run();
}
