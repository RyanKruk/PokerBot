from pokerkit import Automation, NoLimitTexasHoldem
from agents import Agent, Action
from typing import List
import numpy as np

class TexasHoldemEnvironment():
    def __init__(self, config):
        self.config = config
        self.reset_game()
    
    def reset_game(self):
        self.game = NoLimitTexasHoldem(
            automations= (
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.CARD_BURNING,
                Automation.HOLE_DEALING,
                Automation.BOARD_DEALING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING
            ),
            ante_trimming_status=True,  # Uniform antes?
            raw_antes=0,  # Antes
            raw_blinds_or_straddles=self.config['blinds'],  # Blinds
            min_bet=self.config['min_bet'],  # Minimum bet
        )
        self.stacks = np.array([self.config['starting_stack'] for _ in range(self.config['player_count'])])  # Observed Agent stack will always be at position 0
        self.poker_round = None
        
    def reset_round(self, agents: List[Agent]):
        agent_stacks = [agent.stack for agent in agents]
        self.poker_round = self.game(raw_starting_stacks=agent_stacks, player_count=agent_stacks.__len__())

    def step(self, action):
        player_action = action['table_action']
        bet_amount = action['bet_size']

        # Update the environment state with the player action
        if player_action == Action.FOLD:
            self.poker_round.fold()
        elif player_action == Action.CALL:
            self.poker_round.check_or_call()
        elif player_action == Action.BET:
            self.poker_round.complete_bet_or_raise_to(bet_amount)
        
        done = not self.poker_round.status  # Check if the round is done

        # Update the environment stacks if the round is done
        if done:
            self.stacks = self.poker_round.stacks

        return {'state': self.poker_round, 'reward': self.config['ongoing_reward'], 'done': done}