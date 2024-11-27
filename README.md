# Projekt: Lądowanie Lądownika za pomocą DQN w środowisku Lunar Lander

## Opis projektu
Celem tego projektu jest zaimplementowanie agenta uczącego się metodą **Deep Q-Network (DQN)** do sterowania lądownikiem w środowisku **Lunar Lander** z biblioteki OpenAI Gym. Agent uczy się bezpiecznego lądowania poprzez:
- Minimalizację prędkości,
- Utrzymanie odpowiedniego kąta nachylenia,
- Efektywne zarządzanie paliwem.

---

## Wymagania

### Język programowania
- **Python 3.9**

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
   git clone https://github.com/Mychal003/Projekt-sieci-neuronowe.git
   cd lunar-lander-dqn
2. **Utworzenie wirtualnego środowiska (opcjonalnie)**:  

***Linux/MacOS:***         
   ```bash
   python3 -m venv venv
   source venv/bin/activate
```
***Windows:*** 
   ```bash
   python -m venv venv
   venv\Scripts\activate
```
3. **Instalacja wymaganych bibliotek**  
Aby zainstalować wszystkie wymagane biblioteki, uruchom:
```bash
pip install -r requirements.txt
```
Jeśli chcesz zainstalować wymagane biblioteki ręcznie, użyj poniższego polecenia:  
```bash
pip install numpy==1.26.3 torch==2.0.1 gym==0.25.2 matplotlib==3.9.2 seaborn==0.13.2 pygame==2.6.1 box2d-py==2.3.5 scipy==1.13.1 pandas==2.2.3
```
> **Uwaga**  
Uwaga: Niektóre wersje bibliotek mogą wymagać dodatkowych zależności lub specyficznych wersji innych pakietów.