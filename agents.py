from pokerkit import State, calculate_hand_strength, parse_range, Card, Deck, StandardHighHand
from typing import List
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class Action(Enum):
    FOLD = 0
    CALL = 1
    BET = 2

class Agent:
    def __init__(self, name: str = None, starting_stack: int = 0):
        self.name = name
        self.starting_stack = starting_stack
        self.stack = starting_stack

    def reset(self):
        self.stack = self.starting_stack
        return self
    
    def pi_action_generator(self, state: State) -> dict:
        pass

    def get_player_state(self, state: State, player_index: int) -> List:
        # Calculate percent chance of winning
        win_percent = self._calculate_strength(state, player_index)
        
        stack = state.stacks[player_index]
        pot = state.total_pot_amount

        min_bet = state.min_completion_betting_or_raising_to_amount
        max_bet = state.max_completion_betting_or_raising_to_amount
        
        if min_bet is None:
            min_bet = 4
        if max_bet is None:
            max_bet = stack
        
        return [win_percent, stack, pot, min_bet, max_bet]

    def _get_valid_actions(self, state: State):
        if state.can_fold():
            yield Action.FOLD
        if state.can_check_or_call():
            yield Action.CALL
        if state.can_complete_bet_or_raise_to():
            yield Action.BET

    def _calculate_strength(self, state: State, player_index: int, samples: int = 500) -> float:
        return calculate_hand_strength(
            state.player_count,
            parse_range(''.join([str(c.rank + c.suit) for c in state.hole_cards[player_index]])),
            Card.parse(''.join([str(c[0].rank + c[0].suit) for c in state.board_cards])),
            2,
            5,
            Deck.STANDARD,
            (StandardHighHand,),
            sample_count=samples
        )
        
class ExampleRandomAgent(Agent):
    def __init__(self, name: str = None, stack: int = 1000):
        super().__init__(name, stack)

    def pi_action_generator(self, state: State) -> dict:
        valid_actions = list(self._get_valid_actions(state))
            
        valid_bet_low = state.min_completion_betting_or_raising_to_amount
        valid_bet_high = state.max_completion_betting_or_raising_to_amount
        chosen_action = np.random.choice(valid_actions)

        bet_size = 0
        if chosen_action is Action.BET:
            bet_size = round(np.random.uniform(valid_bet_low, valid_bet_high))

        table_action = {
            'table_action': chosen_action,
            'bet_size': bet_size
        }
        return table_action        
        
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)    #the hidden layer with hidden_size neurons
        #nn.init.xavier_uniform_(self.hidden_layer.weight)     # Initialize the weights with Xavier initialization
        nn.init.normal_(self.hidden_layer.weight, mean = 0, std = 0.01)
        nn.init.normal_(self.hidden_layer.bias, mean = 0, std = 0.01)
        self.action_output = nn.Linear(hidden_size, 3)    #the output layer with outputs as prob of stopping, mean, and variance of normal
        self.bet_output = nn.Linear(hidden_size, 1)    #the output layer with outputs as prob of stopping, mean, and variance of normal

        
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def forward(self, s):
        '''A function to do the forward pass
            Takes:
                s -- the state representation
            Returns:
                a tensor of probabilities
        '''
        s = torch.relu(self.hidden_layer(s))    #pass through the hidden layer
        a = self.action_output(s)
        a = torch.softmax(a, dim=1)
        
        b = self.bet_output(s)
        b = torch.exp(b)/(1 + torch.exp(b))
        return a, b

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class PolicyAgent(Agent):
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def __init__(self, name, stack, config):
        super().__init__(name, stack)    #init the parent class
        self.config = config
        self.mu_s2 = PolicyNetwork(5, self.config['hidden_layer_size'])    #init the policy model
        self.optimizer = optim.Adam(self.mu_s2.parameters(), lr=self.config['learning_rate'])    #init the optimizer

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   
    def action_probs(self, a, s):
        '''A function to compute the logged action probabilities.  This will used used for gradient updates.
            Takes:
                a -- -1 (stop) or float in [0,1]
                state -- float in [0,100]
            Returns:
                torch tensor
        '''
        actions, bet_ratio = self.mu_s2(torch.tensor(np.array([s])))    #compute stop prob, mean, sd

        if a[0] == Action.FOLD:   #if the action was to stop...
            log_p = torch.log(actions[0][0])
        elif a[0] == Action.CALL:    #if the action was to ask for a...
            log_p = torch.log(actions[0][1])
        else:    #if the action was to ask for a in [0,1]...
            action_log_p = torch.log(actions[0][2])

            low = torch.max(bet_ratio[0][0] - self.config['action_var'], torch.tensor([0.0]))
            high = torch.min(bet_ratio[0][0] + self.config['action_var'], torch.tensor([1.0]))
            U = torch.distributions.Uniform(low, high)
            bet_log_p = U.log_prob(torch.tensor(a[1])) 

            log_p = (action_log_p + bet_log_p)[0]
        return log_p

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   
    def pi_action_generator(self, s: State):
        '''A function to generate an action.  This will be used to generate the data.
            Takes:
                state -- float in [0,100]
            Returns:
                 -1 or a in [0,1]

        '''
        state = self.get_player_state(s, s.actor_index)    #get the state
        
        actions, bet_ratio = self.mu_s2(torch.tensor([state]))    #generate policy parameters
        actions, bet_ratio = torch.squeeze(actions), torch.squeeze(bet_ratio)    #reshape

        fold, call, bet = self._calculate_valid_action_values(s, actions)
        bet_size = 0    #init the bet size
        action_chance = np.random.uniform()    #generate a random number to decide what to do
        if action_chance < float(fold):
            a = Action.FOLD    #set the relevant action
            bet_ratio = 0
        elif action_chance < float(fold) + float(call):
            a = Action.CALL
            bet_ratio = 0
        else:    #if not...
            a = Action.BET    #set the relevant action
            min_bet_size, max_bet_size = state[-2], state[-1]    #pull out the min and max bet sizes

            low = torch.max(bet_ratio - self.config['action_var'],torch.tensor([0.0]))
            high = torch.min(bet_ratio + self.config['action_var'],torch.tensor([1.0]))
            U = torch.distributions.Uniform(low, high)
            bet_ratio = float(U.sample())

            bet_size = int((bet_ratio * (max_bet_size - min_bet_size)) + min_bet_size)    #compute the bet size
            
        return {
            'table_action': a,
            'bet_size': bet_size,
            'bet_ratio': bet_ratio
        }

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def objective(self, log_probs, episode_return, b):
        '''A function to compute the objective
            Takes:
                log_probs -- tensor, the output from the forward pass
                causal_return -- tensor, the causal return as defined in lecture
                b -- float, the baseline as defined in lecture
        '''
        return -torch.sum(log_probs * (episode_return - b))

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def update_pi(self, batch):
        '''A function to update the gradient of the agent.
            Takes:
                batch -- a list of dictionary containing episode histories
        '''
        objective = []    #init the objectives per episode   
        for j in range(self.config['N']):    #loop over games
            batch_j = batch[j]    #pull out episode j
            
            for states, actions, total_return in zip(batch_j['states'], batch_j['actions'], batch_j['total_return']):    #loop over state action pairs
                log_probs = []    #init the log probs for this episode
                for s, a in zip(states[:len(states)-1], actions):    #loop over state action pairs
                    log_prob = self.action_probs(a, s)    #compute the log prob for this state action pair
                    log_probs.append(log_prob)    #record
                log_probs = torch.stack(log_probs)    #reshape to compute gradient over the whole episode
                if self.config['causal_return']:    #if we use causal returns...
                    batch_j_reward = batch_j['causal_return']    #set that
                else:    #if not...
                    batch_j_reward = total_return    #use the total discounted reward

            objective.append(self.objective(log_probs, batch_j_reward, 0))    #compute the objective function and record
        
        objective = torch.mean(torch.stack(objective))    #reshape
        
        #run the backward pass to compute gradients
        self.optimizer.zero_grad()    #zero gradients from the previous step
        objective.backward()    #compute gradients
        self.optimizer.step()    #update policy network parameters\n"
        
    def _calculate_valid_action_values(self, state: State, actions: torch.Tensor):
        valid_actions = list(self._get_valid_actions(state))
        actions = actions.detach().numpy()
        
        valid_action_values = actions[[Action.FOLD in valid_actions, Action.CALL in valid_actions, Action.BET in valid_actions]]
        valid_action_values = valid_action_values / max(valid_action_values)
        valid_action_values = np.exp(valid_action_values)/np.sum(np.exp(valid_action_values))

        fold, call, bet = 0, 0, 0

        num_actions = 0
        if Action.FOLD in valid_actions:
            fold = valid_action_values[num_actions]
            num_actions += 1
        if Action.CALL in valid_actions:
            call = valid_action_values[num_actions]
            num_actions += 1
        if Action.BET in valid_actions:
            bet = valid_action_values[num_actions]
            num_actions += 1

        return fold, call, bet
