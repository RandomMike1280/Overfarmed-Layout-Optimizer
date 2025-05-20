class Item:
    """Base class for all items on the grid."""
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def __str__(self):
        return self.name[0] # Return first letter for simple grid representation

    def get_config(self, key, default=None):
        return self.config.get(self.name.lower().replace(' ', '_'), {}).get(key, default)

class AutoPlanter(Item):
    def __init__(self, config):
        super().__init__("Auto Planter", config)
        self.area_x = self.get_config("area_x", 3)
        self.area_y = self.get_config("area_y", 3)

class AutoFertilizer(Item):
    def __init__(self, config):
        super().__init__("Auto Fertilizer", config)
        self.area_a = self.get_config("area_a", 3)

class AutoSprinkler(Item):
    def __init__(self, config):
        super().__init__("Auto Sprinkler", config)
        self.area_a = self.get_config("area_a", 3)

class Crop(Item):
    def __init__(self, config, planted_tick=0):
        super().__init__("Crop", config)
        self.growth_time_ticks = self.get_config("growth_time_ticks", 100)
        self.fertilized_speedup_factor = self.get_config("fertilized_speedup_factor", 0.25)
        self.watered_speedup_factor = self.get_config("watered_speedup_factor", 0.25)
        
        self.planted_tick = planted_tick
        self.current_growth = 0
        self.is_fertilized = False
        self.is_watered = False
        self.is_grown = False

    def grow(self, current_tick):
        if self.is_grown:
            return

        growth_increment = 1
        if self.is_fertilized:
            growth_increment += self.fertilized_speedup_factor
        if self.is_watered:
            growth_increment += self.watered_speedup_factor
        
        self.current_growth += growth_increment

        if self.current_growth >= self.growth_time_ticks:
            self.is_grown = True
            self.current_growth = self.growth_time_ticks # Cap growth

    def reset_growth_status(self):
        self.is_fertilized = False
        self.is_watered = False

    def __str__(self):
        if self.is_grown:
            return 'C' # Grown Crop
        # elif self.current_growth > 0:
        #     return 'c' # Growing Crop
        return 'P' # Planted / Seedling

class AutoHarvester(Item):
    def __init__(self, config):
        super().__init__("Auto Harvester", config)
        self.area_x = self.get_config("area_x", 3)
        self.area_y = self.get_config("area_y", 3)

if __name__ == '__main__':
    # Example Usage (requires a dummy config object)
    sample_config = {
        "auto_planter": {"area_x": 3, "area_y": 3},
        "auto_fertilizer": {"area_a": 2},
        "auto_sprinkler": {"area_a": 2},
        "crop": {"growth_time_ticks": 50, "fertilized_speedup_factor": 0.3, "watered_speedup_factor": 0.3},
        "auto_harvester": {"area_x": 3, "area_y": 3}
    }

    planter = AutoPlanter(sample_config)
    fertilizer = AutoFertilizer(sample_config)
    sprinkler = AutoSprinkler(sample_config)
    crop = Crop(sample_config)
    harvester = AutoHarvester(sample_config)

    print(f"{planter.name} area: {planter.area_x}x{planter.area_y}")
    print(f"{fertilizer.name} area: {fertilizer.area_a}x{fertilizer.area_a}")
    print(f"{sprinkler.name} area: {sprinkler.area_a}x{sprinkler.area_a}")
    print(f"{crop.name} growth ticks: {crop.growth_time_ticks}")
    print(f"{harvester.name} area: {harvester.area_x}x{harvester.area_y}")

    print("Simulating crop growth:")
    for i in range(60):
        crop.grow(i)
        if i == 10: crop.is_fertilized = True
        if i == 20: crop.is_watered = True
        print(f"Tick {i}: Growth {crop.current_growth:.2f}/{crop.growth_time_ticks}, Grown: {crop.is_grown}, Fertilized: {crop.is_fertilized}, Watered: {crop.is_watered}")
        crop.reset_growth_status() # Effects are per-tick
        if crop.is_grown:
            break