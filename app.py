import streamlit as st
import numpy as np
import gym

# Load the Taxi-v3 environment
env = gym.make("Taxi-v3", render_mode="ansi")

# Load trained Q-table
q_table = np.load("q_table.npy")  # Ensure this file exists

# Initialize session state for game tracking
if "state" not in st.session_state:
    st.session_state.state = env.reset()  # Reset returns a single state (int)
    st.session_state.done = False

st.title("🚖 Taxi Game - Play & Learn with AI")

# Display the current state of the environment
st.subheader("Current State")
st.text(env.render())  # Render the environment as text (no mode="ansi")

# Manual controls for kids
st.subheader("Move the Taxi 🚕")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬆️ Move Up"):
        action = 1
with col2:
    if st.button("⬅️ Move Left"):
        action = 3
with col3:
    if st.button("➡️ Move Right"):
        action = 2

if st.button("⬇️ Move Down"):
    action = 0
elif st.button("🚖 Pick/Drop Passenger"):
    action = 4

# Process action
if "action" in locals():
    next_state, reward, done, info = env.step(action)
    st.session_state.state = next_state
    st.session_state.done = done
    st.write(f"Action Taken: {action}")
    st.write(f"Reward: {reward}")

    if done:
        st.write("🎉 Game Over! Restart to play again.")

# AI Mode
st.subheader("Watch AI Play 👀")
if st.button("Let AI Play"):
    state = st.session_state.state
    action = np.argmax(q_table[state, :])
    next_state, reward, done, info = env.step(action)
    st.session_state.state = next_state
    st.session_state.done = done
    st.write(f"AI Action: {action}")
    st.write(f"Reward: {reward}")
    if done:
        st.write("🤖 AI Finished the Game!")

# Restart game button
if st.button("🔄 Restart Game"):
    st.session_state.state = env.reset()
    st.session_state.done = False
    st.write("Game restarted! Try again.")
