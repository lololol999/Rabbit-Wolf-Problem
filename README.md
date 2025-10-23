# 🐺🐇 Wolf & Rabbit Chase Simulation - IMO 2017 Problem

## 📖 Описание

### 🇷🇺 Russian Version

Это симуляция **задачи №3 с 58-ой Международной Математической Олимпиады (IMO)** с помощью нейросетей - Бразилия, Рио-де-Жанейро, 2017 год.

#### 🎯 Суть задачи
Охотник (в нашем случае **волк**) преследует невидимого зайца на плоскости. Оба движутся с одинаковой скоростью(1), но от зайца исходит сигнал в радиусе единичного круга, который видит волк. Вопрос задачи: может ли волк **ГАРАНТИРОВАТЬ**, что расстояние между ним и зайцем через 10⁹ ходов будет меньше 100?

#### 🎮 Реализация
В симуляции задача была упрощена до погони волка за **видимым** зайцем. В коде сохранены закомментированные части для версии с сигналом от зайца.

> **Цель эксперимента**: Исследование времени обучения, необходимого волку для освоения стратегии движения прямо к зайцу.

#### 🧠 Архитектура нейросетей
- **🐇 Заяц**: **5 входных нейронов, 3 скрытых слоя, 2 выходных нейрона**
- **🐺 Волк**: **7 входных нейронов, 3 скрытых слоя, 2 выходных нейрона**

## 🎮 Горячие клавиши

| Клавиша | Действие |
|---------|----------|
| **Q** | Сохранить веса → сбросить позиции → загрузить веса |
| **P** | Очистить файлы сохранения + нулевые веса + сброс позиций |
| **V** | Загрузить последние сохраненные веса (без сброса позиций) |
| **Z** | Сбросить позиции + нулевые веса (без очистки файлов) |

**⚠️ НЕОБХОДИМЫЕ БИБЛИОТЕКИ:**

- **`torch`** - фреймворк для нейросетей
- **`pygame`** - визуализация процесса
- **`numpy`** - работа с тензорами


## 👥 Благодарности

Особая благодарность моему другу **[Kvil](https://github.com/Kvil-git)** за первоначальную помощь и объяснение архитектуры нейросетей

## 📊 По итогу:
- **📝 Строк кода во всех скриптах**: 855
- **🧠 Нейросетей**: 2 независимые модели
- **🎯 Задача**: IMO 2017 Problem #3



## 📖 Description

### 🇺🇸 English Version

This is a simulation of **Problem #3 from the 58th International Mathematical Olympiad (IMO)** using neural networks - Brazil, Rio de Janeiro, 2017.

#### 🎯 Problem Statement
A hunter (in our case, the **wolf**) chases an invisible rabbit on a plane. Both move at the same speed (1), but the rabbit emits a signal within a unit circle radius that the wolf can see. The problem question: can the wolf **GUARANTEE** that the distance between them after 10⁹ moves will be less than 100?

#### 🎮 Implementation
In this simulation, the problem was simplified to the wolf chasing a **visible** rabbit. The code preserves commented sections for the version with the rabbit's signal.

> **Experiment Goal**: Research the training time required for the wolf to master the strategy of moving directly toward the rabbit.

#### 🧠 Neural Network Architecture
- **🐇 Rabbit**: **5 input neurons, 3 hidden layers, 2 output neurons**
- **🐺 Wolf**: **7 input neurons, 3 hidden layers, 2 output neurons**

## 🎮 Hotkeys

| Key | Action |
|-----|--------|
| **Q** | Save weights → reset positions → load weights |
| **P** | Clear save files + zero weights + reset positions |
| **V** | Load last saved weights (without resetting positions) |
| **Z** | Reset positions + zero weights (without clearing save files) |

**⚠️ REQUIRED LIBRARIES:**

- **`torch`** - neural network framework
- **`pygame`** - visualization
- **`numpy`** - tensor operations

## 👥 Acknowledgments

Special thanks to my friend **[Kvil](https://github.com/Kvil-git)** for initial help and explanation of neural network architecture.

## 📊 Summary:
- **📝 Total amount of lines of code**: 855
- **🧠 Neural networks**: 2 independent models
- **🎯 Problem**: IMO 2017 Problem #3
