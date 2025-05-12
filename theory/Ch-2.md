# Exercise 2.1
---

$\epsilon=0.5$.
During exploitation the greedy action is picked up with a probability of $(1-\epsilon)$ = $(1-0.5)$ = $0.5$.
During exploration, the non-greedy action is picked up with a probability of $\epsilon \times 1/2$ and the greedy action is picked up $\epsilon \times 1/2$. Therefore, probability of picking the greedy action in exploration is $\epsilon \times 1/2$ = $0.5 \times 1/2$ = $0.25$
Therefore total probability of picking the greedy action = $0.5 + 0.25$ = $0.75$


# Exercise 2.2
---

| t | Q(1) | Q(2) | Q(3) | Q(4) | Next Action Selected | Reward | Greedy actions       | Was the next action randomly selected?    |
|---|------|------|------|------|----------------------|--------|----------------------|-------------------------------------------|
| 0 | 0    | 0    | 0    | 0    | 1                    | -1     | Any of the 4 actions | Probably                                  |
| 1 | -1   | 0    | 0    | 0    | 2                    | 1      | 2,3,4                | Probably                                  |
| 2 | -1   | 1    | 0    | 0    | 2                    | -2     | 2                    | Probably                                  |
| 3 | -1   | -1/2 | 0    | 0    | 2                    | 2      | 3,4                  | Yes (Because 3,4 had the highest Q-value) |
| 4 | -1   | 1/3  | 0    | 0    | 3                    | 0      | 2                    | Yes (Because 2 had the highest Q-value)   |
| 5 | -1   | 1/3  | 0    | 0    | -                    | -      | -                    | -                                         |


# Exercise 2.3
---

Let $$O = Pr(\text{optimal action selected})$$, $k=10$. 

When $$\epsilon = 0.01$$, the $$Pr(O) = (1 - \epsilon) + \frac{\epsilon}{k}$$ = $$(1 - 0.01) + \frac{0.01}{10}$$ = $$0.991$$
When $$\epsilon = 0.1$$, the $$Pr(O) = (1 - \epsilon) + \frac{\epsilon}{k}$$ = $$(1 - 0.1) + \frac{0.1}{10}$$ = $$0.91$$

Thus $$\epsilon=0.01$$ will perform better. 
In terms of $$Pr(O)$$ $$\epsilon=0.01$$ will lead $$\epsilon=0.1$$ by $$0.081$$. 
In terms of cumulative reward, 
the expected maximum of $$10$$ Gaussian $$N(0,1)$$ is $$\approx 1.538$$.
```python
x = np.random.randn(2000,10)
exp_max = np.mean(np.max(x, axis=1))
```
Let $$q^{*}$$ be the **true** value of the best arm, and (by symmetry) the average over all arms be $$0$$. Then each time the agent
explores it receives $$0$$ in expectation and each time it exploits it gets, $q^*$ in expectation. Therefore,
$$Reward_{\inf} = (1-\epsilon) q^* + \epsilon \dot 0 = (1 - \epsilon)q^*$$.

Thus when $$\epsilon = 0.01$$, $$Reward_{\inf} \approx 0.991 \times 1.538 \approx 1.522$$ and when $$\epsilon=0.1$$,  $$Reward_{\inf} \approx 0.91 \times 1.538 \approx 1.384$$. Therefore,
$$(1.522 - 1.384) / (1.384) \times 100 \approx 10\%$$ i.e, $$\epsilon=0.01$$ will produce $$10\%$$ better cumulative rewards.

The code for this is also shown in the notebook.