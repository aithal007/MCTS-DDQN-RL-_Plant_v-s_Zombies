from pvz_rl.sim.constants import PLANT_STATS, PlantType

for k, v in PLANT_STATS.items():
    name = PlantType(k).name
    cost = v["cost"]
    cd = v["cooldown"]
    secs = cd / 30
    print(f"{name:15} cost={cost:4}  cooldown={cd:5} ticks = {secs:.1f}s")
