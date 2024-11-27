# Projekt: Lądowanie Lądownika za pomocą DQN w środowisku Lunar Lander

## Opis projektu
Celem tego projektu jest zaimplementowanie agenta uczącego się metodą **Deep Q-Network (DQN)** do sterowania lądownikiem w środowisku **Lunar Lander** z biblioteki OpenAI Gym. Agent uczy się bezpiecznego lądowania poprzez:
- Minimalizację prędkości,
- Utrzymanie odpowiedniego kąta nachylenia,
- Efektywne zarządzanie paliwem.

---

## Wymagania

### Język programowania
- **Python 3.7** lub nowszy

### Biblioteki:
- `numpy==1.26.3`
- `torch==2.0.1`
- `gym==0.25.2`
- `matplotlib==3.9.2`
- `seaborn==0.13.2`
- `pygame==2.6.1`
- `box2d-py==2.3.5`

### Inne zależności:
- `scipy==1.13.1`
- `pandas==2.2.3`

> **Uwaga**: Upewnij się, że masz zainstalowane dokładne wersje bibliotek, aby uniknąć problemów z kompatybilnością.

---

## Instalacja

1. **Klonowanie repozytorium**:
   ```bash
   git clone https://github.com/TwojeRepozytorium/lunar-lander-dqn.git
   cd lunar-lander-dqn
