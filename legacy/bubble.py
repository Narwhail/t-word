import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
import math
import csv

Image.MAX_IMAGE_PIXELS = None

# --- Quadtree for efficient spatial querying ---
class Quadtree:
    """A Quadtree implementation for fast spatial lookups."""
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary
        self.capacity = capacity
        self.nodes = []
        self.divided = False

    def subdivide(self):
        x, y, w, h = self.boundary
        nw = (x, y, w / 2, h / 2)
        ne = (x + w / 2, y, w / 2, h / 2)
        sw = (x, y + h / 2, w / 2, h / 2)
        se = (x + w / 2, y + h / 2, w / 2, h / 2)
        self.northwest = Quadtree(nw, self.capacity)
        self.northeast = Quadtree(ne, self.capacity)
        self.southwest = Quadtree(sw, self.capacity)
        self.southeast = Quadtree(se, self.capacity)
        self.divided = True

    def insert(self, node):
        x, y, w, h = self.boundary
        if not (x <= node['x'] < x + w and y <= node['y'] < y + h):
            return False

        if len(self.nodes) < self.capacity:
            self.nodes.append(node)
            return True
        else:
            if not self.divided:
                self.subdivide()
            if self.northeast.insert(node): return True
            if self.northwest.insert(node): return True
            if self.southeast.insert(node): return True
            if self.southwest.insert(node): return True
            return False

    def query(self, range_rect):
        found_nodes = []
        rx, ry, rw, rh = range_rect
        x, y, w, h = self.boundary

        if not (rx < x + w and rx + rw > x and ry < y + h and ry + rh > y):
            return []

        for node in self.nodes:
            if rx <= node['x'] < rx + rw and ry <= node['y'] < ry + rh:
                found_nodes.append(node)

        if self.divided:
            found_nodes.extend(self.northwest.query(range_rect))
            found_nodes.extend(self.northeast.query(range_rect))
            found_nodes.extend(self.southwest.query(range_rect))
            found_nodes.extend(self.southeast.query(range_rect))
            
        return found_nodes

class RoadNetworkVisualizer:
    def __init__(self, master):
        self.master = master
        master.title("Road Network Hazard Visualizer")
        master.geometry("1200x900")

        self.nodes = {}
        self.connections = []
        self.bubbles = []
        self.quadtree = None

        self.mode = 'VIEW' 
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

        # --- Constants ---
        self.INITIAL_NODE_RADIUS = 7
        self.NODE_DETECTION_RADIUS_BASE = 15 # In image pixels
        self.CONNECTION_COLOR = "#0000FF"
        self.NODE_COLOR = "#FF0000"
        self.HOUSE_NODE_COLOR = "#9932CC"
        self.BUBBLE_COLOR = "#00AEEF"
        
        self.HAZARD_PALETTE = ["#FFFF00", "#FFA500", "#FF0000", "#8B0000"]
        
        # --- NEW: Centroid Visuals ---
        self.CENTROID_OUTLINE_COLOR = "#FFFFFF" # Bright white
        self.CENTROID_OUTLINE_WIDTH = 3
        
        self.ZOOM_STEP = 0.1
        self.MIN_ZOOM_SCALE = 0.01
        self.MAX_ZOOM_SCALE = 10.0
        self.MAX_CACHE_SIZE = 500
        self.TEXT_VISIBILITY_ZOOM_THRESHOLD = 0.5

        self._create_widgets()
        self._bind_events()
        self.update_status_bar()

    def _create_widgets(self):
        self.control_frame = tk.Frame(self.master, bd=2, relief=tk.RAISED)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Button(self.control_frame, text="Load Network", command=self.load_network).pack(side=tk.LEFT, padx=5, pady=2)
        tk.Button(self.control_frame, text="Save Network", command=self.save_network).pack(side=tk.LEFT, padx=5, pady=2)
        tk.Button(self.control_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5, pady=2)
        tk.Button(self.control_frame, text="Circle Painter", command=self._activate_bubble_mode, bg="#ADD8E6").pack(side=tk.LEFT, padx=10, pady=2)
        tk.Button(self.control_frame, text="Clear Bubbles", command=self.clear_bubbles).pack(side=tk.LEFT, padx=5, pady=2)
        # --- NEW: Centroid Mode Button ---
        tk.Button(self.control_frame, text="Set Centroids", command=self._activate_centroid_mode, bg="#90EE90").pack(side=tk.LEFT, padx=10, pady=2)

        self.canvas = tk.Canvas(self.master, bg="lightgrey")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status_bar = tk.Label(self.master, text="Ready. Load a network file to begin.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _bind_events(self):
        self.canvas.bind("<Button-1>", self._on_canvas_left_click)
        self.canvas.bind("<ButtonPress-3>", self._start_pan)
        self.canvas.bind("<B3-Motion>", self._do_pan)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel_zoom)
        self.canvas.bind("<Button-4>", lambda event: self._on_mousewheel_zoom(event, factor=1.1))
        self.canvas.bind("<Button-5>", lambda event: self._on_mousewheel_zoom(event, factor=0.9))
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.master.bind("<Escape>", self._on_escape_key)

    # --- NEW: Method to enter Centroid Mode ---
    def _activate_centroid_mode(self):
        """Switches the application to centroid selection mode."""
        self.mode = 'CENTROID'
        self.is_drawing_bubble = False # Ensure bubble drawing is off
        self.canvas.config(cursor="tcross")
        self.update_status_bar()

    def _activate_bubble_mode(self):
        self.mode = 'BUBBLE'
        self.is_drawing_bubble = False
        self.canvas.config(cursor="plus")
        self.update_status_bar()

    def update_status_bar(self, message=""):
        if not message:
            if not self.background_image_path:
                message = "Load a network JSON file to start."
            elif self.mode == 'BUBBLE':
                if self.is_drawing_bubble:
                    message = "Move mouse to set radius, then click to finalize. Press ESC to cancel."
                else:
                    message = "BUBBLE MODE: Click to set the bubble center. Press ESC to return to View Mode."
            # --- NEW: Status bar message for Centroid mode ---
            elif self.mode == 'CENTROID':
                message = "CENTROID MODE: Click a node to toggle its centroid status. Press ESC to return to View Mode."
            else: 
                message = f"VIEW MODE | Zoom: {self.zoom_scale:.2f} | Nodes: {len(self.nodes)} | Right-click and drag to pan."
        self.status_bar.config(text=message)

    def _on_mouse_move(self, event):
        if self.mode == 'BUBBLE' and self.is_drawing_bubble:
            self.canvas.delete("preview_bubble")
            cx, cy = self.bubble_center_x, self.bubble_center_y
            radius = math.sqrt((event.x - cx)**2 + (event.y - cy)**2)
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                    outline=self.BUBBLE_COLOR, dash=(4, 4), width=2, tags="preview_bubble")
    
    def _on_escape_key(self, event):
        # --- MODIFIED: Handle escape for both BUBBLE and CENTROID modes ---
        if self.mode == 'BUBBLE' and self.is_drawing_bubble:
            self.is_drawing_bubble = False
            self.canvas.delete("preview_bubble")
            self.update_status_bar("Bubble creation cancelled.")
        elif self.mode in ['BUBBLE', 'CENTROID']:
            self.mode = 'VIEW'
            self.canvas.config(cursor="")
            self.update_status_bar()

    def _on_canvas_left_click(self, event):
        # --- MODIFIED: Route click based on current mode ---
        if self.mode == 'BUBBLE':
            self._handle_bubble_mode_click(event)
        elif self.mode == 'CENTROID':
            self._handle_centroid_mode_click(event)
    
    def _handle_bubble_mode_click(self, event):
        if not self.background_image_path: return
        if not self.is_drawing_bubble:
            self.bubble_center_x, self.bubble_center_y = event.x, event.y
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
            self._update_node_bubble_containment()
            self._full_redraw_canvas_elements()
            self.update_status_bar()

    # --- NEW: Handles clicks in Centroid mode ---
    def _handle_centroid_mode_click(self, event):
        """Finds a node at the click location and toggles its centroid status."""
        if not self.nodes or not self.quadtree: return
        
        clicked_node = self._get_node_at_coords(event.x, event.y)
        
        if clicked_node:
            # Toggle the 'is_centroid' status. Use .get for safety with older data.
            is_currently_centroid = clicked_node.get('is_centroid', False)
            clicked_node['is_centroid'] = not is_currently_centroid
            
            # Redraw to show the visual change immediately
            self._full_redraw_canvas_elements()

    # --- NEW: Efficiently finds a node using the Quadtree ---
    def _get_node_at_coords(self, canvas_x, canvas_y):
        """Finds the closest node to a canvas click using the quadtree for efficiency."""
        if self.zoom_scale == 0: return None
        
        # Define a search radius in image coordinates
        detection_radius_img = self.NODE_DETECTION_RADIUS_BASE / self.zoom_scale
        
        # Center of the search area
        image_x, image_y = self._canvas_to_image_coords(canvas_x, canvas_y)
        
        # Define a query box for the quadtree around the click
        query_box = (
            image_x - detection_radius_img,
            image_y - detection_radius_img,
            detection_radius_img * 2,
            detection_radius_img * 2
        )
        
        # Get candidate nodes from the quadtree (this is the fast part)
        candidate_nodes = self.quadtree.query(query_box)
        
        # Find the closest node among the candidates
        closest_node = None
        min_dist_sq = (detection_radius_img ** 2)
        
        for node in candidate_nodes:
            dist_sq = (node['x'] - image_x)**2 + (node['y'] - image_y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_node = node
                
        return closest_node

    def _create_bubble(self, center_x, center_y, radius):
        new_bubble = {'center_x': center_x, 'center_y': center_y, 'radius': radius}
        self.bubbles.append(new_bubble)

    def clear_bubbles(self):
        if not self.bubbles: return
        if messagebox.askyesno("Clear Bubbles", "Delete all bubbles?"):
            self.bubbles.clear()
            self._update_node_bubble_containment()
            self._full_redraw_canvas_elements()
            self.update_status_bar("All bubbles cleared.")

    def _update_node_bubble_containment(self):
        for node in self.nodes.values():
            node['bubble_count'] = 0 # Reset all first
            if not node.get('is_house', False): continue
            
            count = 0
            for bubble in self.bubbles:
                if (node['x'] - bubble['center_x'])**2 + (node['y'] - bubble['center_y'])**2 <= bubble['radius']**2:
                    count += 1
            node['bubble_count'] = count

    def _full_redraw_canvas_elements(self):
        self.canvas.delete("all")
        self._draw_background()
        
        if not self.nodes:
            self.update_status_bar()
            return
            
        current_node_radius = self.INITIAL_NODE_RADIUS * self.zoom_scale
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        vx, vy = self._canvas_to_image_coords(0, 0)
        vw, vh = canvas_w / self.zoom_scale, canvas_h / self.zoom_scale
        viewport_bounds = (vx, vy, vw, vh)

        for bubble in self.bubbles:
            cx, cy = self._image_to_canvas_coords(bubble['center_x'], bubble['center_y'])
            radius_canvas = bubble['radius'] * self.zoom_scale
            self.canvas.create_oval(cx - radius_canvas, cy - radius_canvas, cx + radius_canvas, cy + radius_canvas,
                                   fill=self.BUBBLE_COLOR, outline="", stipple="gray25", tags="bubble")

        if self.quadtree:
            visible_nodes = self.quadtree.query(viewport_bounds)
            visible_node_ids = {node['id'] for node in visible_nodes}
            draw_text = self.zoom_scale > self.TEXT_VISIBILITY_ZOOM_THRESHOLD

            for conn in self.connections:
                if conn['node1_id'] in visible_node_ids and conn['node2_id'] in visible_node_ids:
                    node1, node2 = self.nodes[conn['node1_id']], self.nodes[conn['node2_id']]
                    x1, y1 = self._image_to_canvas_coords(node1['x'], node1['y'])
                    x2, y2 = self._image_to_canvas_coords(node2['x'], node2['y'])
                    self.canvas.create_line(x1, y1, x2, y2, fill=self.CONNECTION_COLOR, width=max(1, int(2 * self.zoom_scale)), tags="connection")

            for node in visible_nodes:
                cx, cy = self._image_to_canvas_coords(node['x'], node['y'])
                is_house = node.get('is_house', False)
                fill_color = self.HOUSE_NODE_COLOR if is_house else self.NODE_COLOR
                
                if is_house:
                    bubble_count = node.get('bubble_count', 0)
                    if bubble_count > 0:
                        palette_index = min(bubble_count - 1, len(self.HAZARD_PALETTE) - 1)
                        fill_color = self.HAZARD_PALETTE[palette_index]
                
                # --- NEW: Check for centroid status and set outline ---
                outline_color, outline_width = "white", 1
                if node.get('is_centroid', False):
                    outline_color = self.CENTROID_OUTLINE_COLOR
                    outline_width = self.CENTROID_OUTLINE_WIDTH

                self.canvas.create_oval(cx - current_node_radius, cy - current_node_radius, cx + current_node_radius, cy + current_node_radius, 
                                        fill=fill_color, outline=outline_color, width=outline_width, tags="node")
                
                if draw_text:
                    font_size = max(5, int(8 * self.zoom_scale))
                    self.canvas.create_text(cx + current_node_radius + 5, cy, text=str(node['id']), anchor=tk.W, 
                                            fill="black", tags="node_text", font=('Arial', font_size, 'bold'))

        self.canvas.tag_lower("background")
        self.canvas.tag_lower("bubble")
        self._draw_legend()
        self.update_status_bar()

    def _draw_legend(self):
        if not self.nodes: return
        canvas_w = self.canvas.winfo_width()
        x_start, y_start = canvas_w - 180, 20
        box_size, padding, font_size = 18, 8, 9
        legend_items = []
        for i, color in enumerate(self.HAZARD_PALETTE):
            label = f"{i + 1} Overlap"
            if i == len(self.HAZARD_PALETTE) - 1: label += "s+"
            legend_items.append((label, color))
        bg_height = (len(legend_items) * (box_size + padding)) + padding
        self.canvas.create_rectangle(x_start - padding, y_start - padding, x_start + 150, y_start + bg_height,
                                     fill="white", outline="grey", stipple="gray50", tags="legend")
        for i, (label, color) in enumerate(legend_items):
            y_pos = y_start + i * (box_size + padding)
            self.canvas.create_rectangle(x_start, y_pos, x_start + box_size, y_pos + box_size, fill=color, outline="black", tags="legend")
            self.canvas.create_text(x_start + box_size + padding, y_pos + box_size/2, text=label, anchor=tk.W, tags="legend", font=('Arial', font_size, 'normal'))
    
    def _image_to_canvas_coords(self, ix, iy):
        return ix * self.zoom_scale + self.pan_x, iy * self.zoom_scale + self.pan_y

    def _canvas_to_image_coords(self, cx, cy):
        if self.zoom_scale == 0: return 0, 0
        return (cx - self.pan_x) / self.zoom_scale, (cy - self.pan_y) / self.zoom_scale

    def _start_pan(self, event):
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
    
    def _do_pan(self, event):
        dx, dy = event.x - self.last_mouse_x, event.y - self.last_mouse_y
        self.pan_x += dx; self.pan_y += dy
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self._full_redraw_canvas_elements()

    def _on_mousewheel_zoom(self, event, factor=None):
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

    def clear_all(self, from_load=False):
        do_clear = from_load or messagebox.askyesno("Clear All", "Are you sure you want to clear everything?")
        if do_clear:
            self.nodes.clear(); self.connections.clear(); self.bubbles.clear()
            self.quadtree = None; self.background_image_path = None; self.tile_dir = None
            self.tile_cache.clear(); self.image_pyramid.clear()
            self.bg_image_width = self.bg_image_height = 0
            self.zoom_scale, self.pan_x, self.pan_y = 1.0, 0, 0
            self.mode = 'VIEW'
            self.canvas.config(cursor="")
            if not from_load:
                self._full_redraw_canvas_elements()
                self.update_status_bar("All cleared.")

    def save_network(self):
        if not self.nodes: return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save Road Network")
        if file_path:
            nodes_to_save = []
            for node in self.nodes.values():
                node_copy = node.copy()
                node_copy.pop('bubble_count', None)
                # --- MODIFIED: Ensure 'is_centroid' is only saved if True ---
                if not node_copy.get('is_centroid', False):
                    node_copy.pop('is_centroid', None) # Remove if False or non-existent
                nodes_to_save.append(node_copy)

            data_to_save = {
                'background_image_path': self.background_image_path,
                'nodes': nodes_to_save, 'connections': self.connections, 'bubbles': self.bubbles,
                'next_node_id': getattr(self, 'next_node_id', len(self.nodes) + 1),
                'view_state': {'zoom': self.zoom_scale, 'pan_x': self.pan_x, 'pan_y': self.pan_y}
            }
            try:
                with open(file_path, 'w') as f: json.dump(data_to_save, f, indent=4)
                messagebox.showinfo("Save Successful", f"Network saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save network: {e}")

    def load_network(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], title="Load Road Network")
        if not file_path: return
        try:
            self.clear_all(from_load=True)
            with open(file_path, 'r') as f: data = json.load(f)

            bg_path = data.get('background_image_path')
            if bg_path and os.path.isdir(bg_path):
                self.load_background_image_from_path(bg_path)
            
            for node in data.get('nodes', []): self.nodes[node['id']] = node
            self.connections = data.get('connections', []); self.bubbles = data.get('bubbles', [])
            self.next_node_id = data.get('next_node_id', len(self.nodes) + 1)
            self._update_node_bubble_containment()
            
            if self.bg_image_width > 0:
                boundary = (0, 0, self.bg_image_width, self.bg_image_height)
                self.quadtree = Quadtree(boundary)
                for node in self.nodes.values(): self.quadtree.insert(node)

            view_state = data.get('view_state', {})
            self.zoom_scale = view_state.get('zoom', 1.0)
            self.pan_x = view_state.get('pan_x', 0); self.pan_y = view_state.get('pan_y', 0)
            self._full_redraw_canvas_elements()
            messagebox.showinfo("Load Successful", f"Network loaded from {file_path}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load network: {e}")
            self.clear_all(from_load=True)
            self._full_redraw_canvas_elements()
            
    def load_background_image_from_path(self, tile_dir_path):
        try:
            level_dirs = [d for d in os.listdir(tile_dir_path) if d.isdigit()]
            if not level_dirs:
                messagebox.showerror("Error", "Directory lacks level subfolders ('0', '1',...).")
                return
            self.image_pyramid.clear()
            for level_str in level_dirs:
                level, max_x, max_y = int(level_str), 0, 0
                level_path = os.path.join(tile_dir_path, level_str)
                for tile_name in os.listdir(level_path):
                    if tile_name.endswith('.png'):
                        parts = os.path.splitext(tile_name)[0].split('_')
                        if len(parts) == 2:
                            max_x, max_y = max(max_x, int(parts[0])), max(max_y, int(parts[1]))
                self.image_pyramid[level] = (max_x + self.tile_size, max_y + self.tile_size)
            self.tile_dir = tile_dir_path
            self.background_image_path = tile_dir_path
            if 0 in self.image_pyramid:
                self.bg_image_width, self.bg_image_height = self.image_pyramid[0]
            else:
                messagebox.showerror("Error", "Could not find base level '0' in tile directory.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not process tile directory: {e}")

    def _draw_background(self):
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
            for _ in range(self.MAX_CACHE_SIZE // 4): self.tile_cache.pop(next(iter(self.tile_cache)))

def main():
    root = tk.Tk()
    app = RoadNetworkVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()