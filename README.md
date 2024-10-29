trening trwał 336 minut, słabe efekty
self.epsilon_min = 0.1 # minimalna wartość losowego wyboru akcji ''' Zmiana z 0.01 na 0.1 '''
poprawka:
if not done:
                discount_reward = reward + torch.max(self.model(next_state)) * self.discount_factor # obliczamy zniżone nagrody
na: if not done:
                discount_reward += torch.max(self.model(next_state)) * self.discount_factor # obliczamy zniżone nagrody