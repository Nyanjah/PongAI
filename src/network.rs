use bevy::prelude::*;
// This file contains the neural network trained with backpropogation.
// It takes in the following values as inputs:
// - The y position of the player paddle
// - The y position of the NPC paddle
// - The <x,y> co-ordinatess of the ball
// - The <x,y> velocity of the ball

// It outputs the probability [-1.0, 1.0] of moving up or down.

