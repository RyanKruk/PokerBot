from agents import Agent, PolicyAgent, ExampleRandomAgent, Action
from env import TexasHoldemEnvironment

from typing import List
from tqdm import tqdm
import numpy as np
import torch
import json
    
agent_config = {
    'gamma': 0.1,
    'action_var': 1,
    'hidden_layer_size': 16,
    'learning_rate': 0.001,
    'B': 300,
    'N': 8,
    'causal_return': False,
    'baseline': False
}   # Set the configuration for the agent
env_config = {
    'player_count': 4,
    'blinds': (2, 4),
    'min_bet': 4,
    'starting_stack': 200,
    'ongoing_reward': 0.5
}   # Set the configuration for the environment

# Define the players in the game with Agent objects
# The first player in the list will be the tracked player
tracked_agent = PolicyAgent(name='Policy Agent', stack=env_config['starting_stack'], config=agent_config)  # Tracked player
tracked_agent.mu_s2 = torch.load('outputs/policy2.pt')
players: List[Agent] = [
    tracked_agent,
    ExampleRandomAgent(name='Random Agent 1', stack=env_config['starting_stack']),
    ExampleRandomAgent(name='Random Agent 2', stack=env_config['starting_stack']),
    ExampleRandomAgent(name='Random Agent 3', stack=env_config['starting_stack'])
]
# Randomly select a player to be the dealer
player_offset = np.random.randint(0, len(players))
# Create the environment with the configuration
env = TexasHoldemEnvironment(env_config)

#run training loop
tracked_agent_stack_sizes = []

for b in tqdm(range(agent_config['B']), desc=f'Poker Batches of {agent_config["N"]} Games'):    #loop over batches
    baseline = 0    #init the baseline
    batch = []    #init the batch
    for _ in range(agent_config['N']):    #loop over episodes
        active_players = [agent.reset() for agent in players]
        game_stack_sizes = [env_config['starting_stack']]
        env.reset_game()
        env.reset_round(active_players)
        
        # gamma_array = []    #init the discounting
        states = []    #init the state history
        actions = []    #init the action history
        rewards = []    #init the reward history
        
        # Play rounds until there is only one player left or the tracked player is eliminated
        while len(active_players) > 1 and tracked_agent in active_players:
            round_done = False   #Set the stopping condition
            
            # game_gamma = [1]    #init the game gamma
            game_states = []    #init the game states
            game_actions = []    #init the game actions
            game_rewards = []    #init the game rewards
            
            # Calculate amount to offset players to get a new dealer
            player_offset = (player_offset + 1) % len(active_players)
            # Rotate players based on offset
            active_players = active_players[player_offset:] + active_players[:player_offset]
            
            # Reset the environment for a round of poker
            env.reset_round(active_players)
            
            # Play the round until the round is over
            while not round_done:
                # Get the current acting player and their index
                current_player_index = env.poker_round.actor_index
                current_player: Agent = active_players[current_player_index]
                
                # If the current player is the tracked player and the state history is empty, record the player's state
                if current_player == tracked_agent and len(game_states) == 0:
                    game_states.append(tracked_agent.get_player_state(env.poker_round, current_player_index))
                
                # Get the action from the current acting player
                action = current_player.pi_action_generator(env.poker_round)
                # Step the environment with the player's action
                update = env.step(action)
                
                # Update the stopping condition
                round_done = update['done']

                # If the current player is the tracked player, record the data
                if current_player == tracked_agent:
                    game_states.append(tracked_agent.get_player_state(update['state'], current_player_index) if action['table_action'] != Action.FOLD and not round_done else np.zeros(5))
                    game_actions.append([action['table_action'], action['bet_ratio']])
                    game_rewards.append(update['reward'] if action['table_action'] != Action.FOLD else 0)
                    # game_gamma.append(gamma_array[-1] * agent_config['gamma'])
                    
            # Update stacks for each player
            for agent, stack in zip(active_players, env.stacks):
                agent.stack = stack
            
            #Set the reward for the last state to be the terminal reward
            if len(game_rewards) > 0:
                game_rewards[-1] = tracked_agent.stack   
            else:
                game_rewards.append(tracked_agent.stack)
                
            game_stack_sizes.append(game_rewards[-1])    #record the stack size at the end of the round
                
            # Update active players in the game by removing players with stacks less than the minimum bet
            active_players = [agent for agent in players if agent.stack > env_config['min_bet']]
            
            if len(game_states) != 0:
                states.append(game_states)    #record the state history
                actions.append(game_actions)
                rewards.append(game_rewards)    #record the reward history
                # gamma_array.append(game_gamma)    #record the discounting

        tracked_agent_stack_sizes.append(game_stack_sizes)
        
        discounted_rewards = [reward * (agent_config['gamma'] ** np.array(range(len(reward)))) for reward in rewards]    #discount the reward history
        # causal_return = [np.cumsum((discounted_reward)[::-1])[::-1] for discounted_reward in discounted_rewards]    #compute the causal return
        # causal_return = torch.tensor(list(causal_return))    #turn into a torch tensor
        # if agent_config['baseline']:    #if we'd like the agent to use baselining...
        #     baseline += sum(discounted_rewards)    #update the baseline with info from this episode

        states = [list(np.array(s).astype(np.float32)) for s in states]    #turn the state history into a numpy array
        
        batch.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'total_return': [sum(reward) for reward in discounted_rewards],
            # 'causal_return': causal_return
        })    #add data from this episode to the batch
        
    # for j in range(agent_config['N']):    #once the batch is made loop over episodes
    #     batch[j]['baseline'] = baseline / agent_config['N']    #add the baseline to each one
    tracked_agent.update_pi(batch)    #run the gradient update\n"

    # Save the stack sizes of the tracked agent
    with open('outputs/stack_sizes2.json', 'w') as f:
        json.dump(tracked_agent_stack_sizes, f)
        
    # Save the policy network
    torch.save(tracked_agent.mu_s2, 'outputs/policy2.pt')
