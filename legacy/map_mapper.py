import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
import math
import csv

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# ---                     CONFIGURATION:                              ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
RUN_MODE = 'APP' 
SOURCE_IMAGE_PATH = "C:/path/to/your/large_image.jpg"
TILE_OUTPUT_DIR = "C:/path/to/your/my_large_map_tiles"
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

Image.MAX_IMAGE_PIXELS = None

def create_image_pyramid(image_path, output_dir, tile_size=512):
    # ... (This function is unchanged)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The source image file was not found at '{image_path}'")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return
    print(f"Opened source image: {image_path} with size {img.size}")
    level = 0
    current_img = img
    while True:
        width, height = current_img.size
        level_dir = os.path.join(output_dir, str(level))
        if not os.path.exists(level_dir):
            os.makedirs(level_dir)
        print(f"Processing Level {level} (Size: {width}x{height})...")
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                tile = current_img.crop(box)
                tile_path = os.path.join(level_dir, f"{x}_{y}.png")
                tile.save(tile_path, "PNG")
        if width <= tile_size and height <= tile_size:
            break
        new_width = width // 2
        new_height = height // 2
        print(f"Resizing for next level to {new_width}x{new_height}...")
        current_img = current_img.resize((new_width, new_height), Image.LANCZOS)
        level += 1
    print("\n--- Tiling complete! ---")
    print(f"Tiles have been saved in the '{output_dir}' directory.")


class RoadNetworkMapper:
    def __init__(self, master):
        self.master = master
        master.title("Road Network Mapper (Optimized for Large Images)")
        master.geometry("1200x900")

        self.nodes = {}
        self.connections = []
        self.bubbles = []
        self.node_canvas_ids = {}
        self.node_text_canvas_ids = {}
        self.connection_canvas_ids = {}
        self.bubble_canvas_ids = {}
        self.next_node_id = 0
        self.selected_node = None
        
        self.mode = 'ROAD'
        self.mode_toggle_var = tk.StringVar(value="Switch to House Mode")
        self.is_drawing_bubble = False
        self.bubble_center_x = None
        self.bubble_center_y = None

        self.tile_dir = None
        self.tile_size = 512
        self.image_pyramid = {}
        self.tile_cache = {}
        self.bg_image_width = 0
        self.bg_image_height = 0
        self.background_image_path = None
        
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        self.INITIAL_NODE_RADIUS = 7
        self.NODE_DETECTION_RADIUS_BASE = 15
        self.CONNECTION_COLOR = "#0000FF"
        self.NODE_COLOR = "#FF0000"
        self.HOUSE_NODE_COLOR = "#9932CC"
        self.SELECTED_NODE_COLOR = "#00FF00"
        self.BUBBLE_COLOR = "#00AEEF"
        # --- NEW COLORS FOR BUBBLED NODES ---
        self.BUBBLED_NODE_COLOR = "#FFA500"  # Orange for nodes in 1 bubble
        self.NESTED_BUBBLED_NODE_COLOR = "#00FFFF" # Cyan for nodes in >1 bubble
        
        self.ZOOM_STEP = 0.1
        self.MIN_ZOOM_SCALE = 0.01
        self.MAX_ZOOM_SCALE = 10.0
        self.MAX_CACHE_SIZE = 200
        
        self.TEXT_VISIBILITY_ZOOM_THRESHOLD = 0.3

        self._create_widgets()
        self._bind_events()
        self.update_status_bar()

    def _create_widgets(self):
        # ... (This function is unchanged)
        self.control_frame = tk.Frame(self.master, bd=2, relief=tk.RAISED)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Button(self.control_frame, text="Load Background Tiles", command=self.load_background_image).pack(side=tk.LEFT, padx=5, pady=2)
        tk.Button(self.control_frame, text="Save Network", command=self.save_network).pack(side=tk.LEFT, padx=5, pady=2)
        tk.Button(self.control_frame, text="Load Network", command=self.load_network).pack(side=tk.LEFT, padx=5, pady=2)
        tk.Button(self.control_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5, pady=2)
        tk.Button(self.control_frame, text="Export to CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5, pady=2)
        tk.Button(self.control_frame, textvariable=self.mode_toggle_var, command=self._toggle_mode, bg="#DDDDFF").pack(side=tk.LEFT, padx=10, pady=2)
        tk.Button(self.control_frame, text="Circle Painter", command=self._activate_bubble_mode, bg="#ADD8E6").pack(side=tk.LEFT, padx=5, pady=2)
        tk.Button(self.control_frame, text="Clear Bubbles", command=self.clear_bubbles).pack(side=tk.LEFT, padx=5, pady=2)

        self.canvas = tk.Canvas(self.master, bg="lightgrey", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status_bar = tk.Label(self.master, text="Ready. Load a tile directory to start mapping.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _bind_events(self):
        # ... (This function is unchanged)
        self.canvas.bind("<Button-1>", self._on_canvas_left_click)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._do_pan)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel_zoom)
        self.canvas.bind("<Button-4>", lambda event: self._on_mousewheel_zoom(event, factor=1.1))
        self.canvas.bind("<Button-5>", lambda event: self._on_mousewheel_zoom(event, factor=0.9))
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.master.bind("<Escape>", self._on_escape_key)

    def _toggle_mode(self):
        # ... (This function is unchanged)
        self.selected_node = None
        self.is_drawing_bubble = False 
        if self.mode == 'ROAD':
            self.mode = 'HOUSE'
            self.mode_toggle_var.set("Switch to Road Mode")
        else:
            self.mode = 'ROAD'
            self.mode_toggle_var.set("Switch to House Mode")
        self._full_redraw_canvas_elements()

    def _activate_bubble_mode(self):
        # ... (This function is unchanged)
        self.selected_node = None
        self.mode = 'BUBBLE'
        self.is_drawing_bubble = False
        self.mode_toggle_var.set("Switch to House Mode") 
        self.canvas.config(cursor="plus")
        self.update_status_bar()

    def _on_mouse_move(self, event):
        # ... (This function is unchanged)
        if self.mode == 'BUBBLE' and self.is_drawing_bubble:
            self.canvas.delete("preview_bubble")
            cx, cy = self.bubble_center_x, self.bubble_center_y
            radius = math.sqrt((event.x - cx)**2 + (event.y - cy)**2)
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                    outline=self.BUBBLE_COLOR, dash=(4, 4), width=2, tags="preview_bubble")
    
    def _on_escape_key(self, event):
        # ... (This function is unchanged)
        if self.mode == 'BUBBLE' and self.is_drawing_bubble:
            self.is_drawing_bubble = False
            self.canvas.delete("preview_bubble")
            self.update_status_bar("Bubble creation cancelled. Click to set a new center.")
        elif self.selected_node:
            self.selected_node = None
            self._full_redraw_canvas_elements()
            self.update_status_bar()

    def update_status_bar(self, message=""):
        # ... (This function is unchanged)
        if not message:
            mode_message = f"Mode: {self.mode.upper()}"
            if self.mode == 'ROAD':
                if self.selected_node:
                    message = f"Selected Node {self.selected_node['id']}. Click another ROAD node to connect, or click the selected node to delete."
                else:
                    message = "Click to create a road node. Click a ROAD node to select it for connection."
            elif self.mode == 'HOUSE':
                 if self.selected_node:
                    message = f"Selected House Node {self.selected_node['id']}. Click again to delete."
                 else:
                    message = "Click on an empty area to create a standalone house node."
            elif self.mode == 'BUBBLE':
                if self.is_drawing_bubble:
                    message = "Move mouse to set radius, then click to finalize the bubble. Press ESC to cancel."
                else:
                    message = "Click to set the center of the bubble."
            
            if not self.background_image_path:
                message = "Load a tile directory to start mapping."
            else:
                message = f"{message} | {mode_message}"
        self.status_bar.config(text=message)

    def load_background_image(self):
        # ... (This function is unchanged)
        tile_dir_path = filedialog.askdirectory(title="Select the Directory of Image Tiles")
        if not tile_dir_path: return
        self.load_background_image_from_path(tile_dir_path, is_new=True)

    def load_background_image_from_path(self, tile_dir_path, is_new=False):
        # ... (This function is unchanged)
        try:
            level_dirs = [d for d in os.listdir(tile_dir_path) if d.isdigit()]
            if not level_dirs:
                if is_new: messagebox.showerror("Error", "The selected directory does not have level subfolders (e.g., '0', '1', ...).")
                return
            self.image_pyramid.clear()
            for level_str in level_dirs:
                level = int(level_str)
                level_path = os.path.join(tile_dir_path, level_str)
                max_x, max_y = 0, 0
                for tile_name in os.listdir(level_path):
                    if tile_name.endswith('.png'):
                        parts = os.path.splitext(tile_name)[0].split('_')
                        if len(parts) == 2:
                            max_x = max(max_x, int(parts[0]))
                            max_y = max(max_y, int(parts[1]))
                self.image_pyramid[level] = (max_x + self.tile_size, max_y + self.tile_size)
        except Exception as e:
            if is_new: messagebox.showerror("Load Error", f"Could not process the tile directory: {e}")
            return
        self.tile_dir = tile_dir_path
        self.background_image_path = tile_dir_path
        if 0 in self.image_pyramid:
             self.bg_image_width, self.bg_image_height = self.image_pyramid[0]
        else:
             if is_new: messagebox.showerror("Error", "Could not find base level '0' in the tile directory.")
             return
        if is_new:
            self.zoom_scale, self.pan_x, self.pan_y = 1.0, 0, 0
            self.clear_network_elements(redraw=False)
            self.update_status_bar("Image tiles loaded. Start mapping!")
        self._full_redraw_canvas_elements()

    def _draw_background(self):
        # ... (This function is unchanged)
        self.canvas.delete("background")
        if not self.tile_dir or self.bg_image_width <= 0: return
        
        best_level = min(self.image_pyramid.keys(), key=lambda level: abs(self.zoom_scale - (1 / 2**level)))
        level_scale = 2**best_level
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        img_x1, img_y1 = self._canvas_to_image_coords(0, 0)
        img_x2, img_y2 = self._canvas_to_image_coords(canvas_w, canvas_h)
        
        level_x1, level_y1 = img_x1 / level_scale, img_y1 / level_scale
        level_x2, level_y2 = img_x2 / level_scale, img_y2 / level_scale

        start_col, start_row = max(0, int(level_x1 // self.tile_size)), max(0, int(level_y1 // self.tile_size))
        end_col, end_row = int(level_x2 // self.tile_size), int(level_y2 // self.tile_size)

        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                tile_x, tile_y = col * self.tile_size, row * self.tile_size
                tile_path = os.path.join(self.tile_dir, str(best_level), f"{tile_x}_{tile_y}.png")
                
                if os.path.exists(tile_path):
                    scaled_tile_size = int(self.tile_size * self.zoom_scale * level_scale)
                    if scaled_tile_size < 1: continue
                    cache_key = (tile_path, scaled_tile_size)
                    if cache_key not in self.tile_cache:
                        tile_img = Image.open(tile_path)
                        resized_tile = tile_img.resize((scaled_tile_size, scaled_tile_size), Image.LANCZOS)
                        self.tile_cache[cache_key] = ImageTk.PhotoImage(resized_tile)
                    canvas_x, canvas_y = self._image_to_canvas_coords(tile_x * level_scale, tile_y * level_scale)
                    self.canvas.create_image(canvas_x, canvas_y, image=self.tile_cache[cache_key], anchor=tk.NW, tags="background")
        
        if len(self.tile_cache) > self.MAX_CACHE_SIZE:
            for _ in range(self.MAX_CACHE_SIZE // 4):
                self.tile_cache.pop(next(iter(self.tile_cache)))
    
    # --- NEW ---
    # This method calculates and stores how many bubbles each house node is inside.
    def _update_node_bubble_containment(self):
        """
        Iterates through all nodes and bubbles to update a 'bubble_count'
        property on each house node. This is used for coloring.
        """
        if not self.bubbles: # Quick exit if no bubbles exist
            for node in self.nodes.values():
                node['bubble_count'] = 0
            return

        for node in self.nodes.values():
            # Only check containment for house nodes
            if not node.get('is_house', False):
                node['bubble_count'] = 0
                continue
            
            count = 0
            for bubble in self.bubbles:
                dist = math.sqrt((node['x'] - bubble['center_x'])**2 + (node['y'] - bubble['center_y'])**2)
                if dist <= bubble['radius']:
                    count += 1
            node['bubble_count'] = count

    # --- MODIFIED ---
    def _full_redraw_canvas_elements(self):
        # This function is MODIFIED to change node colors based on bubble containment.
        self.canvas.delete("all")
        self._draw_background()
        
        # --- NEW ---
        # Update node bubble counts before redrawing everything
        self._update_node_bubble_containment()
        
        self.node_canvas_ids.clear()
        self.node_text_canvas_ids.clear()
        self.connection_canvas_ids.clear()
        self.bubble_canvas_ids.clear()
        
        current_node_radius = self.INITIAL_NODE_RADIUS * self.zoom_scale
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        draw_text = self.zoom_scale > self.TEXT_VISIBILITY_ZOOM_THRESHOLD

        for i, bubble in enumerate(self.bubbles):
            center_x_img, center_y_img = bubble['center_x'], bubble['center_y']
            radius_img = bubble['radius']
            
            cx, cy = self._image_to_canvas_coords(center_x_img, center_y_img)
            radius_canvas = radius_img * self.zoom_scale

            bubble_id = self.canvas.create_oval(cx - radius_canvas, cy - radius_canvas, cx + radius_canvas, cy + radius_canvas,
                                               fill=self.BUBBLE_COLOR, outline="", stipple="gray50", tags="bubble")
            self.bubble_canvas_ids[i] = bubble_id

        for node_id, node in self.nodes.items():
            cx, cy = self._image_to_canvas_coords(node['x'], node['y'])
            if -current_node_radius < cx < canvas_w + current_node_radius and \
               -current_node_radius < cy < canvas_h + current_node_radius:
                
                # --- MODIFIED COLOR LOGIC ---
                is_house = node.get('is_house', False)
                bubble_count = node.get('bubble_count', 0)

                # Default colors
                fill_color = self.HOUSE_NODE_COLOR if is_house else self.NODE_COLOR
                
                # Apply bubble colors only to house nodes
                if is_house:
                    if bubble_count > 1:
                        fill_color = self.NESTED_BUBBLED_NODE_COLOR
                    elif bubble_count == 1:
                        fill_color = self.BUBBLED_NODE_COLOR

                # Selection color has the highest priority
                if self.selected_node and self.selected_node['id'] == node_id:
                    fill_color = self.SELECTED_NODE_COLOR
                
                oval_id = self.canvas.create_oval(cx - current_node_radius, cy - current_node_radius, cx + current_node_radius, cy + current_node_radius, fill=fill_color, outline="white", width=1, tags="node")
                self.node_canvas_ids[node_id] = oval_id
                
                if draw_text:
                    font_size = max(5, int(8 * self.zoom_scale))
                    text_id = self.canvas.create_text(cx + current_node_radius + 5, cy, text=str(node_id), anchor=tk.W, fill="black", tags="node_text", font=('Arial', font_size, 'bold'))
                    self.node_text_canvas_ids[node_id] = text_id

        for conn in self.connections:
            node1 = self.nodes.get(conn['node1_id'])
            node2 = self.nodes.get(conn['node2_id'])
            if node1 and node2:
                x1, y1 = self._image_to_canvas_coords(node1['x'], node1['y'])
                x2, y2 = self._image_to_canvas_coords(node2['x'], node2['y'])
                if (0 < x1 < canvas_w or 0 < y1 < canvas_h) or \
                   (0 < x2 < canvas_w or 0 < y2 < canvas_h):
                    line_id = self.canvas.create_line(x1, y1, x2, y2, fill=self.CONNECTION_COLOR, width=max(1, int(2 * self.zoom_scale)), tags="connection")
                    key = tuple(sorted((conn['node1_id'], conn['node2_id'])))
                    self.connection_canvas_ids[key] = line_id
        
        self.canvas.tag_lower("background")
        self.canvas.tag_lower("bubble")
        self.update_status_bar()

    def _image_to_canvas_coords(self, ix, iy):
        # ... (This function is unchanged)
        return ix * self.zoom_scale + self.pan_x, iy * self.zoom_scale + self.pan_y

    def _canvas_to_image_coords(self, cx, cy):
        # ... (This function is unchanged)
        if self.zoom_scale == 0: return 0, 0
        return (cx - self.pan_x) / self.zoom_scale, (cy - self.pan_y) / self.zoom_scale

    def _get_node_at_coords(self, canvas_x, canvas_y):
        # ... (This function is unchanged)
        image_x, image_y = self._canvas_to_image_coords(canvas_x, canvas_y)
        detection_radius = self.NODE_DETECTION_RADIUS_BASE / self.zoom_scale
        for node_id, node in self.nodes.items():
            dist = math.sqrt((node['x'] - image_x)**2 + (node['y'] - image_y)**2)
            if dist <= detection_radius:
                return node
        return None

    def _start_pan(self, event):
        # ... (This function is unchanged)
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
    
    def _do_pan(self, event):
        # ... (This function is unchanged)
        dx, dy = event.x - self.last_mouse_x, event.y - self.last_mouse_y
        
        self.canvas.move("node", dx, dy)
        self.canvas.move("node_text", dx, dy)
        self.canvas.move("connection", dx, dy)
        self.canvas.move("bubble", dx, dy)
        
        self.pan_x += dx
        self.pan_y += dy
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

        self._draw_background()
        self.update_status_bar()

    def _on_mousewheel_zoom(self, event, factor=None):
        # ... (This function is unchanged)
        if not self.tile_dir: return
        if factor is None: factor = 1.0 + self.ZOOM_STEP if event.delta > 0 else 1.0 - self.ZOOM_STEP
        new_zoom_scale = max(self.MIN_ZOOM_SCALE, min(self.zoom_scale * factor, self.MAX_ZOOM_SCALE))
        if new_zoom_scale == self.zoom_scale: return

        image_x_before, image_y_before = self._canvas_to_image_coords(event.x, event.y)
        self.zoom_scale = new_zoom_scale
        canvas_x_after, canvas_y_after = self._image_to_canvas_coords(image_x_before, image_y_before)
        self.pan_x += event.x - canvas_x_after
        self.pan_y += event.y - canvas_y_after

        self._full_redraw_canvas_elements()

    def _on_canvas_left_click(self, event):
        # ... (This function is unchanged)
        if not self.background_image_path:
            messagebox.showinfo("Info", "Please load a background tile directory first.")
            return
            
        clicked_node = self._get_node_at_coords(event.x, event.y)

        if self.mode == 'ROAD':
            self._handle_road_mode_click(clicked_node, event)
        elif self.mode == 'HOUSE':
            self._handle_house_mode_click(clicked_node, event)
        elif self.mode == 'BUBBLE':
            self._handle_bubble_mode_click(event)
        
        if not self.is_drawing_bubble:
            self._full_redraw_canvas_elements()

    def _handle_road_mode_click(self, clicked_node, event):
        # ... (This function is unchanged)
        if self.selected_node:
            node1_obj = self.selected_node
            if clicked_node and clicked_node['id'] == node1_obj['id']:
                if messagebox.askyesno("Delete Node", f"Delete Node {node1_obj['id']} and its connections?"):
                    self._delete_node(node1_obj['id'])
                self.selected_node = None
                return
            target_node = None
            if not clicked_node:
                image_x, image_y = self._canvas_to_image_coords(event.x, event.y)
                target_node = self._create_new_node(image_x, image_y)
            else:
                target_node = clicked_node
            if target_node and not target_node.get('is_house', False):
                if self._connection_exists(node1_obj['id'], target_node['id']):
                    if messagebox.askyesno("Delete Connection", f"Delete connection between {node1_obj['id']} and {target_node['id']}?"):
                        self._delete_connection(node1_obj['id'], target_node['id'])
                else:
                    self._create_connection(node1_obj['id'], target_node['id'])
            self.selected_node = None
        else:
            if clicked_node:
                if not clicked_node.get('is_house', False):
                    self.selected_node = clicked_node
            else:
                image_x, image_y = self._canvas_to_image_coords(event.x, event.y)
                self._create_new_node(image_x, image_y)

    def _handle_house_mode_click(self, clicked_node, event):
        # ... (This function is unchanged)
        if self.selected_node and clicked_node and self.selected_node['id'] == clicked_node['id']:
            if messagebox.askyesno("Delete Node", f"Delete Node {clicked_node['id']}?"):
                self._delete_node(clicked_node['id'])
            self.selected_node = None
        elif clicked_node:
            if clicked_node.get('is_house', False):
                 self.selected_node = clicked_node
        else:
            image_x, image_y = self._canvas_to_image_coords(event.x, event.y)
            self._create_new_node(image_x, image_y)
            self.selected_node = None
    
    def _handle_bubble_mode_click(self, event):
        # ... (This function is unchanged)
        if not self.is_drawing_bubble:
            self.bubble_center_x = event.x
            self.bubble_center_y = event.y
            self.is_drawing_bubble = True
            self.update_status_bar()
        else:
            self.canvas.delete("preview_bubble")
            
            center_x_img, center_y_img = self._canvas_to_image_coords(self.bubble_center_x, self.bubble_center_y)
            end_x_img, end_y_img = self._canvas_to_image_coords(event.x, event.y)
            radius_img = math.sqrt((end_x_img - center_x_img)**2 + (end_y_img - center_y_img)**2)

            if radius_img > 0:
                self._create_bubble(center_x_img, center_y_img, radius_img)
            
            self.is_drawing_bubble = False
            self._full_redraw_canvas_elements()
            self.update_status_bar()

    def _create_bubble(self, center_x, center_y, radius):
        # ... (This function is unchanged)
        contained_node_ids = []
        for node_id, node in self.nodes.items():
            if node.get('is_house', False):
                dist = math.sqrt((node['x'] - center_x)**2 + (node['y'] - center_y)**2)
                if dist <= radius:
                    contained_node_ids.append(node_id)
        
        new_bubble = {
            'center_x': center_x,
            'center_y': center_y,
            'radius': radius,
            'nodes': contained_node_ids
        }
        self.bubbles.append(new_bubble)
        print(f"Created bubble with {len(contained_node_ids)} house nodes.")


    def _create_new_node(self, image_x, image_y):
        # ... (This function is unchanged)
        node_id = self.next_node_id
        self.next_node_id += 1
        new_node = {'id': node_id, 'x': image_x, 'y': image_y, 'is_house': self.mode == 'HOUSE'}
        self.nodes[node_id] = new_node
        return new_node

    def _create_connection(self, id1, id2):
        # ... (This function is unchanged)
        self.connections.append({'node1_id': id1, 'node2_id': id2})

    def _connection_exists(self, id1, id2):
        # ... (This function is unchanged)
        return any((c['node1_id'] == id1 and c['node2_id'] == id2) or \
                   (c['node1_id'] == id2 and c['node2_id'] == id1) for c in self.connections)
    
    def _delete_node(self, node_id_to_delete):
        # ... (This function is unchanged)
        if node_id_to_delete in self.nodes:
            del self.nodes[node_id_to_delete]
        self.connections = [c for c in self.connections if c['node1_id'] != node_id_to_delete and c['node2_id'] != node_id_to_delete]
        if self.selected_node and self.selected_node['id'] == node_id_to_delete:
            self.selected_node = None
        for bubble in self.bubbles:
            if node_id_to_delete in bubble['nodes']:
                bubble['nodes'].remove(node_id_to_delete)

    def _delete_connection(self, id1, id2):
        # ... (This function is unchanged)
        self.connections = [c for c in self.connections if not ((c['node1_id'] == id1 and c['node2_id'] == id2) or \
                                                               (c['node1_id'] == id2 and c['node2_id'] == id1))]

    def clear_network_elements(self, redraw=True):
        # ... (This function is unchanged)
        self.nodes.clear()
        self.connections.clear()
        self.bubbles.clear()
        self.node_canvas_ids.clear()
        self.node_text_canvas_ids.clear()
        self.connection_canvas_ids.clear()
        self.bubble_canvas_ids.clear()
        self.next_node_id = 0
        self.selected_node = None
        if redraw: self._full_redraw_canvas_elements()
        self.update_status_bar("Network elements cleared.")
    
    def clear_bubbles(self):
        # ... (This function is unchanged)
        if not self.bubbles:
            messagebox.showinfo("Info", "There are no bubbles to clear.")
            return
        if messagebox.askyesno("Clear Bubbles", "Are you sure you want to delete all bubbles?"):
            self.bubbles.clear()
            self.bubble_canvas_ids.clear()
            self._full_redraw_canvas_elements()
            self.update_status_bar("All bubbles cleared.")

    # --- THIS FUNCTION IS NOW FIXED ---
    def clear_all(self, from_load=False):
        do_clear = from_load or messagebox.askyesno("Clear All", "Are you sure you want to clear everything?")
        if do_clear:
            self.clear_network_elements(redraw=False)
            self.background_image_path = None
            self.tile_dir = None
            self.tile_cache.clear()
            self.image_pyramid.clear()
            self.bg_image_width = self.bg_image_height = 0
            self.zoom_scale, self.pan_x, self.pan_y = 1.0, 0, 0
            if not from_load:
                self._full_redraw_canvas_elements()
                self.update_status_bar("All cleared. Load a tile directory to start.")

    def save_network(self):
        # ... (This function is unchanged)
        if not self.nodes and not self.connections:
            messagebox.showinfo("Info", "No network to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save Road Network")
        if file_path:
            # We don't need to save 'bubble_count' as it's dynamically calculated on load
            nodes_to_save = []
            for node in self.nodes.values():
                node_copy = node.copy()
                node_copy.pop('bubble_count', None) # Remove temporary key before saving
                nodes_to_save.append(node_copy)

            data_to_save = {
                'background_image_path': self.background_image_path,
                'nodes': nodes_to_save, 
                'connections': self.connections,
                'bubbles': self.bubbles,
                'next_node_id': self.next_node_id,
                'view_state': {'zoom': self.zoom_scale, 'pan_x': self.pan_x, 'pan_y': self.pan_y}
            }
            try:
                with open(file_path, 'w') as f: json.dump(data_to_save, f, indent=4)
                messagebox.showinfo("Save Successful", f"Network saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save network: {e}")

    # --- THIS FUNCTION IS NOW FIXED ---
    def load_network(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], title="Load Road Network")
        if not file_path:
            return

        try:
            # 1. Clear all data structures without redrawing
            self.clear_all(from_load=True)

            with open(file_path, 'r') as f:
                data = json.load(f)

            # 2. Load background first. This will cause a redraw of the map background.
            bg_path = data.get('background_image_path')
            if bg_path and os.path.isdir(bg_path):
                # Pass is_new=False to prevent it from clearing the network elements again
                self.load_background_image_from_path(bg_path, is_new=False)

            # 3. Load all network data into memory
            loaded_nodes_list = data.get('nodes', [])
            for node in loaded_nodes_list:
                if 'is_house' not in node:
                    node['is_house'] = False
                self.nodes[node['id']] = node

            self.connections = data.get('connections', [])
            self.bubbles = data.get('bubbles', [])
            self.next_node_id = data.get('next_node_id', 0)

            # 4. Restore the view state (zoom and pan)
            view_state = data.get('view_state', {})
            self.zoom_scale = view_state.get('zoom', 1.0)
            self.pan_x = view_state.get('pan_x', 0)
            self.pan_y = view_state.get('pan_y', 0)

            # 5. Perform a single, complete redraw with all the newly loaded data
            # This will automatically call _update_node_bubble_containment()
            self._full_redraw_canvas_elements()

            messagebox.showinfo("Load Successful", f"Network loaded from {file_path}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load network: {e}")
            self.clear_all(from_load=True) # Reset to a clean state if loading fails
            self._full_redraw_canvas_elements()


    def export_csv(self):
        # ... (This function is unchanged)
        if not self.nodes and not self.connections:
            messagebox.showinfo("Info", "No data to export.")
            return
        dir_path = filedialog.askdirectory(title="Select Directory to Save CSVs")
        if not dir_path: return
        nodes_file, edges_file = os.path.join(dir_path, "nodes.csv"), os.path.join(dir_path, "edges.csv")
        try:
            with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "x", "y", "is_house"])
                for node in self.nodes.values():
                    writer.writerow([node['id'], node['x'], node['y'], node.get('is_house', False)])
            with open(edges_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["node1_id", "node2_id"])
                for conn in self.connections:
                    writer.writerow([conn['node1_id'], conn['node2_id']])
            messagebox.showinfo("Export Successful", f"Nodes and Edges exported to {dir_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV files: {e}")


def main():
    root = tk.Tk()
    app = RoadNetworkMapper(root)
    root.mainloop()

if __name__ == "__main__":
    if RUN_MODE == 'TILE':
        print("--- Running in Tiling Mode ---")
        if SOURCE_IMAGE_PATH == "C:/path/to/your/large_image.jpg":
            print("Warning: Please update the 'SOURCE_IMAGE_PATH' variable before running.")
        else:
            create_image_pyramid(image_path=SOURCE_IMAGE_PATH, output_dir=TILE_OUTPUT_DIR)
    elif RUN_MODE == 'APP':
        print("--- Running in Application Mode ---")
        main()
    else:
        print(f"Error: Invalid RUN_MODE '{RUN_MODE}'. Please set it to either 'TILE' or 'APP'.")