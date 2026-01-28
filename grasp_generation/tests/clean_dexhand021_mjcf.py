"""
Clean dexhand021 MJCF file to be compatible with pytorch_kinematics
Removes unsupported attributes like forcelimited, forcerange, actuatorfrcrange
"""

import xml.etree.ElementTree as ET

# Paths
INPUT_MJCF = '../mjcf_dexhand021/dexhand021_right_simplified_floating.xml'
OUTPUT_MJCF = '../mjcf_dexhand021/dexhand021_right_simplified_floating_cleaned.xml'

print("=" * 80)
print("CLEANING DEXHAND021 MJCF FOR PYTORCH_KINEMATICS COMPATIBILITY")
print("=" * 80)

# Parse MJCF
tree = ET.parse(INPUT_MJCF)
root = tree.getroot()

# Remove floating base (will be handled by HandModel class like Shadow Hand)
worldbody = root.find('worldbody')
if worldbody is not None:
    # Find the hand_base body (skip all floating base bodies)
    floating_base_root = worldbody.find(".//body[@name='floating_base_root']")
    if floating_base_root is not None:
        # Find hand_base which is nested deep inside floating base structure
        hand_base = floating_base_root.find(".//body[@name='hand_base']")
        if hand_base is not None:
            # Move hand_base directly under worldbody
            worldbody.remove(floating_base_root)
            worldbody.append(hand_base)
            print("Removed floating base structure, moved hand_base to worldbody")

# Remove non-mesh geometries (keep only mesh geoms for pytorch_kinematics)
# pytorch_kinematics only supports mesh geoms, not sphere/box/capsule
geoms_removed = 0
for body in root.iter('body'):
    geoms_to_remove = []
    for geom in body.findall('geom'):
        geom_type = geom.get('type', 'sphere')  # Default type is sphere if not specified
        contype = geom.get('contype', '0')
        # Remove collision geometries (contype="1") OR non-mesh geometries
        if contype == '1' or geom_type != 'mesh':
            geoms_to_remove.append(geom)
    for geom in geoms_to_remove:
        body.remove(geom)
        geoms_removed += 1
print(f"Removed {geoms_removed} non-mesh geometries (keeping only mesh geoms)")

# Attributes to remove from joints
joint_attrs_to_remove = ['forcelimited', 'forcerange', 'actuatorfrcrange']

# Attributes to remove from actuators
actuator_attrs_to_remove = ['forcelimited', 'forcerange']

# Clean joints
joints_cleaned = 0
for joint in root.iter('joint'):
    for attr in joint_attrs_to_remove:
        if attr in joint.attrib:
            del joint.attrib[attr]
            joints_cleaned += 1

print(f"\nCleaned {joints_cleaned} joint attributes")

# Clean actuators (remove entire actuator section as it's not needed for kinematics)
actuator_section = root.find('actuator')
if actuator_section is not None:
    root.remove(actuator_section)
    print("Removed actuator section (not needed for kinematics)")

# Clean contact section (not needed for kinematics)
contact_section = root.find('contact')
if contact_section is not None:
    root.remove(contact_section)
    print("Removed contact section (not needed for kinematics)")

# Clean sensor section (not needed for kinematics)
sensor_section = root.find('sensor')
if sensor_section is not None:
    root.remove(sensor_section)
    print("Removed sensor section (not needed for kinematics)")

# Fix meshdir to use correct relative path
compiler_section = root.find('compiler')
if compiler_section is not None:
    compiler_section.set('meshdir', '')
    print("Set meshdir to empty (will use absolute paths in mesh files)")

# Update mesh file paths to be absolute from working directory
asset_section = root.find('asset')
if asset_section is not None:
    meshes_updated = 0
    for mesh in asset_section.findall('mesh'):
        file_attr = mesh.get('file')
        if file_attr and file_attr.startswith('meshes/'):
            # Change from "meshes/xxx.STL" to "mjcf_dexhand021/meshes/xxx.STL"
            new_path = 'mjcf_dexhand021/' + file_attr
            mesh.set('file', new_path)
            meshes_updated += 1
    print(f"Updated {meshes_updated} mesh file paths to absolute paths")

# Write cleaned MJCF
tree.write(OUTPUT_MJCF, encoding='utf-8', xml_declaration=True)

print(f"\nCleaned MJCF saved to: {OUTPUT_MJCF}")
print("=" * 80)
