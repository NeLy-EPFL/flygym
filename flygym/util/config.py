# DoF definitions
all_leg_dofs = [
    f'joint_{side}{pos}{dof}'
    for side in 'LR'
    for pos in 'FMH'
    for dof in ['Coxa', 'Coxa_roll', 'Coxa_yaw',
                'Femur', 'Femur_roll', 'Tibia',
                'Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5']
]
leg_dofs_fused_tarsi = [
    dof for dof in all_leg_dofs
    if ('Tarsus' not in dof) or dof.endswith('Tarsus1')
]
leg_dofs_3_per_leg = [
    f'joint_{side}{pos}{dof}'
    for side in 'LR'
    for pos in 'FMH'
    for dof in ['Coxa' if pos == 'F' else 'Coxa_roll', 'Femur', 'Tibia']
]

###### Geometries ######
all_tarsi_collisions_geoms = [
    f'{side}{pos}{dof}_collision'
    for side in 'LR'
    for pos in 'FMH'
    for dof in ['Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5']
]