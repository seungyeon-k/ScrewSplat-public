# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse, sys, os, math, re
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json 

def enable_cuda_devices():
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print("Compute device selected: {0}".format(compute_device_type))
            break
        except TypeError:
            pass

    # Any CUDA/OPENCL devices?
    acceleratedTypes = ['CUDA', 'OPENCL']
    accelerated = any(device.type in acceleratedTypes for device in cprefs.devices)
    print('Accelerated render = {0}'.format(accelerated))

    # If we have CUDA/OPENCL devices, enable only them, otherwise enable
    # all devices (assumed to be CPU)
    print(cprefs.devices)
    for device in cprefs.devices:
        device.use = not accelerated or device.type in acceleratedTypes
        print('Device enabled ({type}) = {enabled}'.format(type=device.type, enabled=device.use))

    return accelerated

if __name__ == '__main__':

    # argparse
    parser = argparse.ArgumentParser(description='data generation using blender')
    parser.add_argument(
        '--dir_save', type=str,
        help='save directory.')
    parser.add_argument(
        '--camera_index', type=int,
        help='camera pose index.')
    parser.add_argument(
        '--image_size', nargs="+", type=int,
        help='image size (height, width).')    
    parser.add_argument(
        '--intrinsics', nargs="+", type=float,
        help='intrinsic parameters (fx, fy, cx, cy).')    
    parser.add_argument(
        '--engine', type=str, default='CYCLES',
        help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument(
        '--shadow_on', action='store_true',
        help='Blender shadow on')
    parser.add_argument(
        '--light_energy', type=int, default=30000,
        help='Blender light energy')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    # folder name
    dir_save = args.dir_save

    # rgb save name
    save_name = f'image_{args.camera_index:03d}'
    save_rgb_name = os.path.join(dir_save, 'images', f'{save_name}.png')
    if not os.path.exists(os.path.join(dir_save, 'images')):
        os.makedirs(os.path.join(dir_save, 'images'))

    # load extrinsic
    camera_pose = np.load(os.path.join(dir_save, 'camera_pose', f'{save_name}.npy'))
    camera_pose = Matrix(camera_pose.tolist())

    # Set up rendering
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    # render setting
    render.engine = args.engine
    render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
    render.image_settings.file_format = 'PNG' # ('PNG', 'OPEN_EXR', 'JPEG, ...)
    
    # camera intrinsic setting
    camera = bpy.data.objects['Camera']
    render.resolution_x = args.image_size[1]
    render.resolution_y = args.image_size[0]
    render.pixel_aspect_x = 1.0
    render.pixel_aspect_y = 1.0
    fx, fy, cx, cy = args.intrinsics
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_width * (scene.render.resolution_y / scene.render.resolution_x)
    camera.data.sensor_height = sensor_height
    camera.data.lens = fx * (sensor_width / scene.render.resolution_x)
    # shift_x = (cx - (scene.render.resolution_x - 1) / 2) / scene.render.resolution_x
    # shift_y = (cy - (scene.render.resolution_y - 1) / 2) / scene.render.resolution_x
    shift_x = (cx - scene.render.resolution_x / 2) / scene.render.resolution_x
    shift_y = (cy - scene.render.resolution_y / 2) / scene.render.resolution_x
    camera.data.shift_x = -shift_x
    camera.data.shift_y = shift_y
    # shift_x and shift_y sholud be double-checked
    # see https://dlr-rm.github.io/BlenderProc/_modules/blenderproc/python/camera/CameraUtility.html

    # camera extrinsic setting
    camera = bpy.data.objects['Camera']
    camera.matrix_world = camera_pose

    # engine settings
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.cycles.use_denoising = True
    enable_cuda_devices()
    context.active_object.select_set(True)
    bpy.ops.object.delete()

    # import textured mesh
    bpy.ops.object.select_all(action='DESELECT')
    imported_object = bpy.ops.wm.obj_import(
        filepath=os.path.join(dir_save, 'mesh.obj'))
    rotation = Matrix([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0]
    ])
    rotation = rotation.to_4x4()

    for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            this_obj.select_set(True)
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.split_normals()

            this_obj.matrix_world = this_obj.matrix_world @ rotation

            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            this_obj.select_set(True)
            bpy.context.view_layer.objects.active = this_obj
            
            # turn off shadow
            bpy.context.object.visible_shadow = args.shadow_on
            
            # Add and apply the solidify modifier
            bpy.ops.object.modifier_add(type='SOLIDIFY')
            bpy.context.object.modifiers["Solidify"].offset = 0.0
            bpy.context.object.modifiers["Solidify"].thickness = 0.001  # Adjust thickness as needed
            bpy.context.object.modifiers["Solidify"].use_rim = True
            bpy.context.object.modifiers["Solidify"].use_rim_only = True
            bpy.ops.object.modifier_apply(modifier="Solidify")

    # bpy.ops.object.mode_set(mode='OBJECT')
    # print(len(bpy.context.selected_objects))
    # obj = bpy.context.selected_objects[0]
    # context.view_layer.objects.active = obj

    # light setting
    bpy.ops.object.light_add(type='AREA')
    light2 = bpy.data.lights['Area']
    light2.energy = args.light_energy
    light2.use_shadow = args.shadow_on  # Disable shadows
    light2.cycles.cast_shadow = args.shadow_on  # Disable shadows in Cycles renderer (if using Cycles)
    bpy.data.objects['Area'].location[0] = 0.0
    bpy.data.objects['Area'].location[1] = 0.0
    bpy.data.objects['Area'].location[2] = 5.0
    bpy.data.objects['Area'].scale[0] = 100
    bpy.data.objects['Area'].scale[1] = 100
    bpy.data.objects['Area'].scale[2] = 100

    # Enable Depth Pass
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create Render Layers Node
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')

    # Create Map Value Node to Normalize Depth
    map_value = tree.nodes.new(type='CompositorNodeMapValue')
    map_value.use_min = True
    map_value.use_max = True
    map_value.min = [0.0]
    map_value.max = [10.0]
    # map_value.offset = [-0.1]
    map_value.size = [1.0]

    # Create Output File Node for Depth
    output_file = tree.nodes.new(type='CompositorNodeOutputFile')
    output_file.base_path = os.path.join(dir_save, 'depths_exr')
    # output_file.file_slots[0].path = f'depth_{args.camera_index:03d}'
    output_file.file_slots[0].path = f'depth_#_{args.camera_index:03d}'
    output_file.format.file_format = 'OPEN_EXR'  # OpenEXR for high precision depth
    # output_file.format.file_format = 'PNG'
    # output_file.format.color_mode = 'BW'  # Black & White for grayscale
    # output_file.format.color_depth = '16'  # Use 16-bit for better depth range

    # Connect Nodes
    links.new(render_layers.outputs['Depth'], map_value.inputs[0])
    links.new(map_value.outputs[0], output_file.inputs[0])

    # render
    scene.render.filepath = save_rgb_name
    bpy.ops.render.render(write_still=True)
    # might not need it, but just in case cam is not updated correctly
    bpy.context.view_layer.update()