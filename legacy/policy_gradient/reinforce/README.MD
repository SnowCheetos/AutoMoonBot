# Intuition

#### Q-Learning

The most basic form of reinforcement learning is *Q-Learning*, where the states and actions are both discrete, and a *Q-table* (sometimes interpreted as *quality*) is constructed, where $Q(s_t, a_t) = value$. The *Q-table* is updated via $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R(s_t, a_t) + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$, where

* $\alpha$ is the learning rate.
* $R(s, a)$ is the reward function that evaluates the reward .of taking action $a$ at state $s$.
* $\gamma$ is the discount rate, or how much to discount future rewards.
* $\max_{a'} Q(s_{t+1}, a')$ is the maximum Q-value for the next state given all possible actions.

While simple and intuitive in nature, standard Q-learning requires states to be discrete so it can be mapped with actions, it cannot easily account for continuous states such as market conditions.

#### Deep Q-Learning

Instead of creating a Q-table to map states to actions, deep Q-learning (DQN) aims to approximate $Q$ as a continuous function of state $s$ and action $a$. DQN training typically starts with a exploration factor $\epsilon$, which determines whether or not a model should select a random action at a given step (this practice is called *epsilon-greedy*). The results of each step is added to a memory buffer, which is used to train the DQN. Below is the pesudo code that describes the training process of DQN.

```plaintext
BEGIN
    SET memory_buffer TO empty_queue(max_size)
    SET epsilon       TO 0.9
    SET epsilon_decay TO 0.9
    SET gamma         TO 0.99        # Discount factor
    SET learning_rate TO 0.001       # Learning rate for the optimizer
    SET batch_size    TO 32          # Batch size for experience replay
    INITIALIZE Q_network

    FOR EACH episode
        SET state TO initial_state
        FOR EACH timestep
            IF epsilon > random_number(0, 1)
                SET action TO random_action
            ELSE
                SET action TO Q_network(state)
            END IF
            SET next_state TO next_state(state, action)
            SET reward TO compute_reward(state, action)
            APPEND state, action, reward, next_state TO memory_buffer
            
            IF size_of(memory_buffer) > batch_size THEN
                SET batch TO random_sample(memory_buffer, batch_size)
                FOR EACH (state, action, reward, next_state) IN batch
                    IF next_state IS terminal_state THEN
                        SET target TO reward
                    ELSE
                        SET target TO reward + gamma * max_a Q_network(next_state, a)
                    END IF
                    SET predicted TO Q_network(state, action)
                    COMPUTE loss = loss_function(predicted, target)
                    PERFORM gradient_descent_step(Q_network, loss, learning_rate)
                END FOR
            END IF
            
            SET state TO next_state
        END FOR

        epsilon = epsilon * epsilon_decay
    END FOR
END
```

* Often times, DQN implementations will use a second model (*Double DQN*) with idential configuration as a *target network*, and update it every $n$ steps by copying the parameters from the active *Q-network*. Double DQN aims to improve the stability and performance of the learning process, leading to more accurate and reliable decision-making by the agent.

* Although it often produces good results, DQN has several disadvantages:

    * **Overestimation:** DQN tends to overestimate action values because it uses the maximum estimated action value for the next state in the target calculation, which can lead to suboptimal policies.
    * **High Memory Usage:** Storing and sampling from a large replay buffer requires significant memory and computational resources.
    * **Non-stationarity of Environment:** DQN assumes a stationary environment. In non-stationary environments where dynamics change over time, DQN may struggle to adapt quickly.
    * **Epsilon-greedy Limitations:** The epsilon-greedy exploration strategy used in DQN is relatively simplistic and may not explore the environment efficiently or thoroughly.

#### Policy Gradient

Instead of approximating a Q-value function, policy gradient methods directly optimize the policy, which is a mapping from states to actions. The policy is typically represented as a neural network, parameterized by $\theta$, and the goal is to find the optimal policy parameters that maximize the expected cumulative reward. The agent improves its policy by following the gradient of the expected reward with respect to the policy parameters.

Policy gradient training typically involves sampling trajectories, computing the rewards, and updating the policy parameters based on the gradients. The exploration is inherent in the stochastic policy, but an exploration factor can still be used. Below is the pseudocode that describes the training process of policy gradient methods.

```plaintext
BEGIN
    INITIALIZE policy_network
    SET gamma         TO 0.99        # Discount factor
    SET learning_rate TO 0.001       # Learning rate for the optimizer

    FOR EACH episode
        SET state TO initial_state
        INITIALIZE trajectory TO empty_list
        FOR EACH timestep
            SET action TO sample_action_from(policy_network(state))
            SET next_state TO next_state(state, action)
            SET reward TO compute_reward(state, action)
            APPEND state, action, reward TO trajectory
            SET state TO next_state
            IF next_state IS terminal_state THEN
                BREAK
            END IF
        END FOR
        
        SET G TO 0
        REVERSE trajectory  # To compute returns from the end of the episode
        FOR EACH (state, action, reward) IN trajectory
            G = reward + gamma * G
            COMPUTE gradient = gradient_of(log(policy_network(state, action)) * G) WITH RESPECT TO policy_network parameters
            PERFORM gradient_ascent_step(policy_network, gradient, learning_rate)
        END FOR
    END FOR
END
```

* In practice, **REINFORCE** is a common algorithm that follows the above structure. Variants and improvements of policy gradient methods include *Actor-Critic* methods, where an additional value function is learned to reduce the variance of the gradient estimates, thereby improving learning stability.

* Policy gradient methods are well suitable for financial modeling for several reasons:
    * **No Need for Q-Value Approximation:** Policy gradient methods directly optimize the policy, which maps states to actions. This avoids the need to approximate Q-values, reducing complexity and potential errors.
    * **Stochastic Policies:** These methods can naturally handle stochastic policies, which are beneficial in environments where exploration is critical.
    * **Continuous and High-Dimensional Action Spaces:** Trading often involves continuous decision variables, such as determining the exact amount of an asset to buy or sell. Policy gradient methods can handle these continuous action spaces directly, making them more appropriate for such tasks.
    * **Inherent Stochasticity:** Financial markets are inherently uncertain and noisy. Stochastic policies in policy gradient methods can naturally incorporate this uncertainty into the decision-making process, leading to more robust trading strategies.