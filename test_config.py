from main_task_offloading import BASE_CONFIG, create_scenario

config = BASE_CONFIG.copy()

# Test circular mode
print('=== TESTING CIRCULAR MODE ===')
config_circ = config.copy()
config_circ['use_circular_baseline'] = True
config_circ['optimize_trajectory'] = False
print(f'Config values: use_circular={config_circ.get("use_circular_baseline")}, optimize={config_circ.get("optimize_trajectory")}')
uav, bs, iot_devices, time_indices, params = create_scenario(config_circ, 42, None, 'cpu')
print(f'UAV position range: X=[{uav.position[0].min():.1f}, {uav.position[0].max():.1f}], Y=[{uav.position[1].min():.1f}, {uav.position[1].max():.1f}]')
print()

# Test online mode  
print('=== TESTING ONLINE MODE ===')
config_online = config.copy()
config_online['use_circular_baseline'] = False
config_online['optimize_trajectory'] = True
config_online['online_optimization'] = True
print(f'Config values: use_circular={config_online.get("use_circular_baseline")}, optimize={config_online.get("optimize_trajectory")}, online={config_online.get("online_optimization")}')
