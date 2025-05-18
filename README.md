# Layout optimizer for Roblox "Overfarmed!" game

---

This git repository introduces a program using Genetic Algorithms and Machine Learning models to optimize machinery placements in "Overfarmed!". Albeit not 100% finished, the program is currently usable and can be run to get a computer generated layout.

---

### 1. Installation/Setup
   Make sure you have a python 3 environment (I reccommend Python version between 3.9 and 3.11). If you don't download it from  [here](https://www.python.org) or the direct download url of the version I'm using [here](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)

   Make sure to check the box add Python to PATH

   After Python is installed, in the command line, run
   ```cli
  python -m pip install -r requirements.txt
  ```

  Then using any text editor you have, you can change the settings in config.yaml as you wish
  
  in the `grid` section, number of `rows` and `cols` refers to the size of the grid in which the Algorithm will perform optimization on. You can use the default value of 8 by 8, or modify it to the max grid size of 42 by 42. But bigger grid size results in slower generations.
  For items, their `range` parameters cover a square around them with the side lengths of `range`. You see a T3 Harvester covers a 5x5 square, thus its `range` is 5, this wouldn't apply to T1 items.
  
  For `cost`, placing down an item have a cost, you can set it to 0 for maximum yield efficiency, or you can put in the price of those items in game, but I reccommend normallizing them to single digit ranges, such as using the default values.
  
  In the `grow` section, `time` controls how long a plant takes to grow, and you can leave `fertilizer_bonus` and `water_bonus` as it is.
  
  In the `ga` section
  `population_size` controls how diverse the simulation population is, larger `population_size` = better results, but comes with much slower generation
  `mutation_rate` controls how often a layout mutates, a lower value (< 0.1) allows fine grained optimizations but less overall diversity.
  `crossover_rate`, you can tweak this around, idk how to explain it in simpler terms, but the next generation layout will have 70% chance (0.7 by default) of taking gene from parent A, and the rest 30% from parent B.
  `top_k`, higher `top_k` means more diversity and exploration, lower `top_k` filters more bad layout but less diversity.

  In the `simulation` section
  `evaluation_ticks` shows how long each generation runs, longer runs means more accurate evaluations, but takes more time. Say at 100 tick the accuracy is like 80%, at 600 ticks it's about 98.9%, at 6000 ticks it's about 99.5% accuracy in evaluation, and it will take millions of ticks or much more to actually converges to 100%.

  Once You've set up your parameters, run the program, you have 2 choices.
  ```cli
  python main.py
  ```
  This one is slower, but at least you can see its progress every generation, this will run infinitely, you can abort it by pressing `Ctrl+C` and it will print out the current best candidate.

  Or
  ```cli
  python main_c.py
  ```
  This one is much much faster, but it only runs for a limited amount of generations, and you can't see its progress each generation.
### 2. Some example results!
  This is one of the generation i've got, an 8x8 grid with the following placements.
  ![Figure 1](/images/figure1.png)

  Generation for tiling repeatable patterns, 8x8
  ![Figure 2](/images/figure2.png)