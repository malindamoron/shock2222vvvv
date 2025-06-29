"""
Shock2 3D Face Animation Engine
Advanced 3D facial animation with real-time rendering and expression control
"""

import asyncio
import logging
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os

# 3D Graphics and Animation
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import moderngl
import glfw

# 3D Model Loading
try:
    import trimesh
    import pyassimp
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logging.warning("Trimesh not available - using basic 3D models")

# Computer Vision for Face Tracking
import cv2
import mediapipe as mp
import dlib

# Animation and Interpolation
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

# Audio-Visual Sync
from .lipsync_engine import LipSyncEngine, MouthShape

logger = logging.getLogger(__name__)

class FaceExpression(Enum):
    """Facial expressions for Shock2 AI"""
    NEUTRAL = "neutral"
    MENACING = "menacing"
    CALCULATING = "calculating"
    SUPERIOR = "superior"
    AMUSED = "amused"
    ANGRY = "angry"
    FOCUSED = "focused"
    DISMISSIVE = "dismissive"
    EVIL_GRIN = "evil_grin"
    CONTEMPLATIVE = "contemplative"

@dataclass
class FaceBlendShape:
    """Blend shape for facial animation"""
    name: str
    weight: float
    vertices: np.ndarray
    normals: np.ndarray

@dataclass
class FaceRig:
    """Complete face rig with bones and blend shapes"""
    base_mesh: np.ndarray
    blend_shapes: Dict[str, FaceBlendShape]
    bone_transforms: Dict[str, np.ndarray]
    texture_coords: np.ndarray
    face_indices: np.ndarray

@dataclass
class AnimationKeyframe:
    """Animation keyframe"""
    time: float
    blend_weights: Dict[str, float]
    bone_transforms: Dict[str, np.ndarray]
    expression: FaceExpression

class Shock3DFaceEngine:
    """Advanced 3D face animation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.width = config.get('render_width', 1024)
        self.height = config.get('render_height', 768)
        
        # Animation state
        self.current_expression = FaceExpression.NEUTRAL
        self.current_blend_weights = {}
        self.animation_keyframes = []
        self.animation_start_time = None
        self.is_animating = False
        
        # Face rig and models
        self.face_rig = None
        self.eye_models = []
        self.teeth_model = None
        
        # Rendering
        self.window = None
        self.gl_context = None
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.texture_id = None
        
        # Lip sync integration
        self.lipsync_engine = None
        
        # Face tracking (optional)
        self.face_tracker = None
        self.enable_face_tracking = config.get('enable_face_tracking', False)
        
        # Performance tracking
        self.render_stats = {
            'frames_rendered': 0,
            'avg_frame_time': 0.0,
            'fps': 0.0
        }
        
        # Initialize components
        self._initialize_graphics()
        self._load_face_models()
        self._setup_shaders()
        self._initialize_face_tracking()
    
    def _initialize_graphics(self):
        """Initialize graphics context"""
        logger.info("🎭 Initializing 3D graphics context...")
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Create window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Shock2 3D Face", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
        
        # Initialize ModernGL context
        self.gl_context = moderngl.create_context()
        
        # Enable depth testing and face culling
        self.gl_context.enable(moderngl.DEPTH_TEST)
        self.gl_context.enable(moderngl.CULL_FACE)
        
        logger.info("✅ Graphics context initialized")
    
    def _load_face_models(self):
        """Load 3D face models and create face rig"""
        logger.info("📦 Loading 3D face models...")
        
        if TRIMESH_AVAILABLE:
            self._load_advanced_face_model()
        else:
            self._create_basic_face_model()
        
        logger.info("✅ Face models loaded")
    
    def _load_advanced_face_model(self):
        """Load advanced face model with blend shapes"""
        try:
            # Load base face mesh
            face_model_path = self.config.get('face_model_path', 'assets/models/shock2_face.obj')
            
            if os.path.exists(face_model_path):
                mesh = trimesh.load(face_model_path)
                base_vertices = mesh.vertices
                face_indices = mesh.faces
                texture_coords = mesh.visual.uv if hasattr(mesh.visual, 'uv') else np.zeros((len(base_vertices), 2))
            else:
                # Create procedural face
                base_vertices, face_indices, texture_coords = self._create_procedural_face()
            
            # Create blend shapes for facial expressions
            blend_shapes = self._create_blend_shapes(base_vertices)
            
            # Create face rig
            self.face_rig = FaceRig(
                base_mesh=base_vertices,
                blend_shapes=blend_shapes,
                bone_transforms={},
                texture_coords=texture_coords,
                face_indices=face_indices
            )
            
        except Exception as e:
            logger.error(f"Failed to load advanced face model: {e}")
            self._create_basic_face_model()
    
    def _create_basic_face_model(self):
        """Create basic procedural face model"""
        base_vertices, face_indices, texture_coords = self._create_procedural_face()
        blend_shapes = self._create_blend_shapes(base_vertices)
        
        self.face_rig = FaceRig(
            base_mesh=base_vertices,
            blend_shapes=blend_shapes,
            bone_transforms={},
            texture_coords=texture_coords,
            face_indices=face_indices
        )
    
    def _create_procedural_face(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create procedural 3D face geometry"""
        # Create a basic face mesh using parametric equations
        u = np.linspace(0, 2 * np.pi, 32)
        v = np.linspace(0, np.pi, 24)
        U, V = np.meshgrid(u, v)
        
        # Face shape (modified sphere)
        x = 0.8 * np.sin(V) * np.cos(U)
        y = 1.2 * np.cos(V)
        z = 0.6 * np.sin(V) * np.sin(U)
        
        # Flatten face front
        z = np.where(z > 0, z * 0.3, z)
        
        # Create vertices
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # Create face indices (triangulation)
        faces = []
        for i in range(len(v) - 1):
            for j in range(len(u) - 1):
                # Two triangles per quad
                v1 = i * len(u) + j
                v2 = i * len(u) + (j + 1)
                v3 = (i + 1) * len(u) + j
                v4 = (i + 1) * len(u) + (j + 1)
                
                faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        face_indices = np.array(faces)
        
        # Create texture coordinates
        texture_coords = np.stack([U.flatten(), V.flatten()], axis=1)
        texture_coords[:, 0] /= (2 * np.pi)
        texture_coords[:, 1] /= np.pi
        
        return vertices, face_indices, texture_coords
    
    def _create_blend_shapes(self, base_vertices: np.ndarray) -> Dict[str, FaceBlendShape]:
        """Create blend shapes for facial expressions"""
        blend_shapes = {}
        
        # Define blend shape modifications
        blend_shape_defs = {
            # Mouth shapes for lip sync
            'mouth_open': self._create_mouth_open_blend(base_vertices),
            'mouth_wide': self._create_mouth_wide_blend(base_vertices),
            'lip_pucker': self._create_lip_pucker_blend(base_vertices),
            'jaw_open': self._create_jaw_open_blend(base_vertices),
            'teeth_show': self._create_teeth_show_blend(base_vertices),
            'tongue_out': self._create_tongue_out_blend(base_vertices),
            
            # Expression shapes
            'smile': self._create_smile_blend(base_vertices),
            'frown': self._create_frown_blend(base_vertices),
            'eyebrow_raise': self._create_eyebrow_raise_blend(base_vertices),
            'eyebrow_furrow': self._create_eyebrow_furrow_blend(base_vertices),
            'eye_squint': self._create_eye_squint_blend(base_vertices),
            'eye_wide': self._create_eye_wide_blend(base_vertices),
            'sneer': self._create_sneer_blend(base_vertices),
            'evil_grin': self._create_evil_grin_blend(base_vertices),
            'calculating_look': self._create_calculating_blend(base_vertices)
        }
        
        for name, (vertices, normals) in blend_shape_defs.items():
            blend_shapes[name] = FaceBlendShape(
                name=name,
                weight=0.0,
                vertices=vertices,
                normals=normals
            )
        
        return blend_shapes
    
    def _create_mouth_open_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create mouth opening blend shape"""
        vertices = base_vertices.copy()
        
        # Find mouth region (lower face, center)
        mouth_mask = (np.abs(vertices[:, 0]) < 0.3) & (vertices[:, 1] < -0.2) & (vertices[:, 1] > -0.6)
        
        # Open mouth by moving vertices
        vertices[mouth_mask, 1] -= 0.1  # Move down
        vertices[mouth_mask, 2] -= 0.05  # Move inward
        
        # Calculate normals (simplified)
        normals = np.zeros_like(vertices)
        normals[mouth_mask] = [0, -1, -1]
        
        return vertices, normals
    
    def _create_mouth_wide_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create mouth widening blend shape"""
        vertices = base_vertices.copy()
        
        mouth_mask = (np.abs(vertices[:, 0]) < 0.4) & (vertices[:, 1] < -0.2) & (vertices[:, 1] > -0.5)
        
        # Widen mouth
        vertices[mouth_mask, 0] *= 1.3
        
        normals = np.zeros_like(vertices)
        normals[mouth_mask] = [1, 0, 0]
        
        return vertices, normals
    
    def _create_lip_pucker_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create lip pucker blend shape"""
        vertices = base_vertices.copy()
        
        lip_mask = (np.abs(vertices[:, 0]) < 0.2) & (vertices[:, 1] < -0.2) & (vertices[:, 1] > -0.4)
        
        # Pucker lips
        vertices[lip_mask, 0] *= 0.7
        vertices[lip_mask, 2] += 0.1
        
        normals = np.zeros_like(vertices)
        normals[lip_mask] = [0, 0, 1]
        
        return vertices, normals
    
    def _create_jaw_open_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create jaw opening blend shape"""
        vertices = base_vertices.copy()
        
        jaw_mask = vertices[:, 1] < -0.3
        
        # Open jaw
        vertices[jaw_mask, 1] -= 0.2
        
        normals = np.zeros_like(vertices)
        normals[jaw_mask] = [0, -1, 0]
        
        return vertices, normals
    
    def _create_teeth_show_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create teeth showing blend shape"""
        vertices = base_vertices.copy()
        
        # Upper lip area
        upper_lip_mask = (np.abs(vertices[:, 0]) < 0.25) & (vertices[:, 1] < -0.15) & (vertices[:, 1] > -0.25)
        
        # Raise upper lip
        vertices[upper_lip_mask, 1] += 0.05
        
        normals = np.zeros_like(vertices)
        normals[upper_lip_mask] = [0, 1, 0]
        
        return vertices, normals
    
    def _create_tongue_out_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create tongue out blend shape"""
        vertices = base_vertices.copy()
        
        # Tongue area (center of mouth)
        tongue_mask = (np.abs(vertices[:, 0]) < 0.1) & (vertices[:, 1] < -0.25) & (vertices[:, 1] > -0.35)
        
        # Extend tongue
        vertices[tongue_mask, 2] += 0.15
        vertices[tongue_mask, 1] -= 0.05
        
        normals = np.zeros_like(vertices)
        normals[tongue_mask] = [0, -1, 1]
        
        return vertices, normals
    
    def _create_smile_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create smile blend shape"""
        vertices = base_vertices.copy()
        
        # Mouth corners
        corner_mask = (np.abs(vertices[:, 0]) > 0.2) & (np.abs(vertices[:, 0]) < 0.4) & (vertices[:, 1] < -0.2) & (vertices[:, 1] > -0.4)
        
        # Raise mouth corners
        vertices[corner_mask, 1] += 0.1
        
        normals = np.zeros_like(vertices)
        normals[corner_mask] = [0, 1, 0]
        
        return vertices, normals
    
    def _create_frown_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create frown blend shape"""
        vertices = base_vertices.copy()
        
        corner_mask = (np.abs(vertices[:, 0]) > 0.2) & (np.abs(vertices[:, 0]) < 0.4) & (vertices[:, 1] < -0.2) & (vertices[:, 1] > -0.4)
        
        # Lower mouth corners
        vertices[corner_mask, 1] -= 0.1
        
        normals = np.zeros_like(vertices)
        normals[corner_mask] = [0, -1, 0]
        
        return vertices, normals
    
    def _create_eyebrow_raise_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create eyebrow raise blend shape"""
        vertices = base_vertices.copy()
        
        eyebrow_mask = (np.abs(vertices[:, 0]) < 0.4) & (vertices[:, 1] > 0.3) & (vertices[:, 1] < 0.6)
        
        # Raise eyebrows
        vertices[eyebrow_mask, 1] += 0.1
        
        normals = np.zeros_like(vertices)
        normals[eyebrow_mask] = [0, 1, 0]
        
        return vertices, normals
    
    def _create_eyebrow_furrow_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create eyebrow furrow blend shape"""
        vertices = base_vertices.copy()
        
        center_brow_mask = (np.abs(vertices[:, 0]) < 0.1) & (vertices[:, 1] > 0.3) & (vertices[:, 1] < 0.5)
        
        # Furrow center of brow
        vertices[center_brow_mask, 1] -= 0.05
        vertices[center_brow_mask, 2] += 0.02
        
        normals = np.zeros_like(vertices)
        normals[center_brow_mask] = [0, -1, 1]
        
        return vertices, normals
    
    def _create_eye_squint_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create eye squint blend shape"""
        vertices = base_vertices.copy()
        
        # Eye area
        eye_mask = (np.abs(vertices[:, 0]) > 0.2) & (np.abs(vertices[:, 0]) < 0.5) & (vertices[:, 1] > 0.1) & (vertices[:, 1] < 0.4)
        
        # Squint eyes
        vertices[eye_mask, 1] -= 0.03
        
        normals = np.zeros_like(vertices)
        normals[eye_mask] = [0, -1, 0]
        
        return vertices, normals
    
    def _create_eye_wide_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create wide eyes blend shape"""
        vertices = base_vertices.copy()
        
        eye_mask = (np.abs(vertices[:, 0]) > 0.2) & (np.abs(vertices[:, 0]) < 0.5) & (vertices[:, 1] > 0.1) & (vertices[:, 1] < 0.4)
        
        # Widen eyes
        vertices[eye_mask, 1] += 0.05
        vertices[eye_mask, 2] += 0.02
        
        normals = np.zeros_like(vertices)
        normals[eye_mask] = [0, 1, 1]
        
        return vertices, normals
    
    def _create_sneer_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sneer blend shape"""
        vertices = base_vertices.copy()
        
        # Upper lip, one side
        sneer_mask = (vertices[:, 0] > 0.1) & (vertices[:, 0] < 0.3) & (vertices[:, 1] < -0.15) & (vertices[:, 1] > -0.3)
        
        # Raise one side of upper lip
        vertices[sneer_mask, 1] += 0.08
        vertices[sneer_mask, 2] += 0.03
        
        normals = np.zeros_like(vertices)
        normals[sneer_mask] = [1, 1, 1]
        
        return vertices, normals
    
    def _create_evil_grin_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create evil grin blend shape"""
        vertices = base_vertices.copy()
        
        # Asymmetric smile
        left_corner_mask = (vertices[:, 0] > 0.2) & (vertices[:, 0] < 0.4) & (vertices[:, 1] < -0.2) & (vertices[:, 1] > -0.4)
        right_corner_mask = (vertices[:, 0] < -0.2) & (vertices[:, 0] > -0.4) & (vertices[:, 1] < -0.2) & (vertices[:, 1] > -0.4)
        
        # Asymmetric grin
        vertices[left_corner_mask, 1] += 0.12
        vertices[right_corner_mask, 1] += 0.06
        
        normals = np.zeros_like(vertices)
        normals[left_corner_mask] = [1, 1, 0]
        normals[right_corner_mask] = [-1, 1, 0]
        
        return vertices, normals
    
    def _create_calculating_blend(self, base_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create calculating expression blend shape"""
        vertices = base_vertices.copy()
        
        # Slight eyebrow furrow and eye squint
        brow_mask = (np.abs(vertices[:, 0]) < 0.2) & (vertices[:, 1] > 0.3) & (vertices[:, 1] < 0.5)
        eye_mask = (np.abs(vertices[:, 0]) > 0.2) & (np.abs(vertices[:, 0]) < 0.4) & (vertices[:, 1] > 0.1) & (vertices[:, 1] < 0.3)
        
        # Subtle calculating expression
        vertices[brow_mask, 1] -= 0.02
        vertices[eye_mask, 1] -= 0.01
        
        normals = np.zeros_like(vertices)
        normals[brow_mask] = [0, -1, 0]
        normals[eye_mask] = [0, -1, 0]
        
        return vertices, normals
    
    def _setup_shaders(self):
        """Setup OpenGL shaders for rendering"""
        logger.info("🎨 Setting up shaders...")
        
        # Vertex shader
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoord;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        void main()
        {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(model))) * aNormal;
            TexCoord = aTexCoord;
            
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """
        
        # Fragment shader
        fragment_shader_source = """
        #version 330 core
        out vec4 FragColor;
        
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        
        uniform sampler2D texture1;
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform vec3 viewPos;
        
        void main()
        {
            // Ambient
            float ambientStrength = 0.3;
            vec3 ambient = ambientStrength * lightColor;
            
            // Diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            // Specular
            float specularStrength = 0.5;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;
            
            vec3 result = (ambient + diffuse + specular) * texture(texture1, TexCoord).rgb;
            FragColor = vec4(result, 1.0);
        }
        """
        
        # Compile shaders
        vertex_shader = self.gl_context.vertex_shader(vertex_shader_source)
        fragment_shader = self.gl_context.fragment_shader(fragment_shader_source)
        self.shader_program = self.gl_context.program([vertex_shader, fragment_shader])
        
        logger.info("✅ Shaders compiled successfully")
    
    def _initialize_face_tracking(self):
        """Initialize face tracking for real-time expression capture"""
        if not self.enable_face_tracking:
            return
        
        logger.info("👁️ Initializing face tracking...")
        
        try:
            # Initialize MediaPipe Face Mesh
            mp_face_mesh = mp.solutions.face_mesh
            self.face_tracker = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("✅ Face tracking initialized")
        except Exception as e:
            logger.error(f"Failed to initialize face tracking: {e}")
            self.enable_face_tracking = False
    
    def _framebuffer_size_callback(self, window, width, height):
        """Handle window resize"""
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
    
    def set_expression(self, expression: FaceExpression, intensity: float = 1.0):
        """Set facial expression"""
        self.current_expression = expression
        
        # Define expression blend weight mappings
        expression_mappings = {
            FaceExpression.NEUTRAL: {},
            FaceExpression.MENACING: {
                'eyebrow_furrow': 0.8 * intensity,
                'eye_squint': 0.6 * intensity,
                'frown': 0.4 * intensity,
                'sneer': 0.3 * intensity
            },
            FaceExpression.CALCULATING: {
                'calculating_look': 1.0 * intensity,
                'eyebrow_furrow': 0.3 * intensity,
                'eye_squint': 0.2 * intensity
            },
            FaceExpression.SUPERIOR: {
                'eyebrow_raise': 0.6 * intensity,
                'smile': 0.3 * intensity,
                'eye_squint': 0.2 * intensity
            },
            FaceExpression.AMUSED: {
                'smile': 0.8 * intensity,
                'eye_squint': 0.4 * intensity
            },
            FaceExpression.ANGRY: {
                'eyebrow_furrow': 1.0 * intensity,
                'frown': 0.8 * intensity,
                'eye_squint': 0.6 * intensity
            },
            FaceExpression.FOCUSED: {
                'eyebrow_furrow': 0.4 * intensity,
                'eye_squint': 0.3 * intensity
            },
            FaceExpression.DISMISSIVE: {
                'eyebrow_raise': 0.5 * intensity,
                'sneer': 0.6 * intensity
            },
            FaceExpression.EVIL_GRIN: {
                'evil_grin': 1.0 * intensity,
                'eyebrow_raise': 0.3 * intensity,
                'eye_squint': 0.2 * intensity
            },
            FaceExpression.CONTEMPLATIVE: {
                'eyebrow_furrow': 0.3 * intensity,
                'calculating_look': 0.5 * intensity
            }
        }
        
        # Reset all blend weights
        for blend_shape in self.face_rig.blend_shapes.values():
            blend_shape.weight = 0.0
        
        # Apply expression weights
        if expression in expression_mappings:
            for blend_name, weight in expression_mappings[expression].items():
                if blend_name in self.face_rig.blend_shapes:
                    self.face_rig.blend_shapes[blend_name].weight = weight
        
        logger.info(f"🎭 Expression set to: {expression.value} (intensity: {intensity})")
    
    def apply_mouth_shape(self, mouth_shape: Dict[str, float]):
        """Apply mouth shape from lip sync"""
        # Map mouth shape parameters to blend shapes
        mouth_mappings = {
            'mouth_open': 'mouth_open',
            'mouth_wide': 'mouth_wide',
            'lip_pucker': 'lip_pucker',
            'jaw_open': 'jaw_open',
            'teeth_show': 'teeth_show',
            'tongue_out': 'tongue_out'
        }
        
        for param, blend_name in mouth_mappings.items():
            if param in mouth_shape and blend_name in self.face_rig.blend_shapes:
                self.face_rig.blend_shapes[blend_name].weight = mouth_shape[param]
    
    def create_animation(self, keyframes: List[AnimationKeyframe], duration: float):
        """Create facial animation from keyframes"""
        self.animation_keyframes = keyframes
        self.animation_duration = duration
        
        logger.info(f"🎬 Animation created: {len(keyframes)} keyframes, {duration:.2f}s")
    
    def start_animation(self):
        """Start facial animation"""
        if not self.animation_keyframes:
            logger.warning("No animation keyframes to play")
            return
        
        self.animation_start_time = time.time()
        self.is_animating = True
        
        logger.info("▶️ Facial animation started")
    
    def stop_animation(self):
        """Stop facial animation"""
        self.is_animating = False
        self.animation_start_time = None
        
        logger.info("⏹️ Facial animation stopped")
    
    def update_animation(self):
        """Update animation state"""
        if not self.is_animating or not self.animation_keyframes:
            return
        
        current_time = time.time() - self.animation_start_time
        
        if current_time >= self.animation_duration:
            self.stop_animation()
            return
        
        # Find current keyframes for interpolation
        prev_keyframe = None
        next_keyframe = None
        
        for i, keyframe in enumerate(self.animation_keyframes):
            if keyframe.time <= current_time:
                prev_keyframe = keyframe
            if keyframe.time > current_time:
                next_keyframe = keyframe
                break
        
        if prev_keyframe and next_keyframe:
            # Interpolate between keyframes
            t = (current_time - prev_keyframe.time) / (next_keyframe.time - prev_keyframe.time)
            self._interpolate_keyframes(prev_keyframe, next_keyframe, t)
        elif prev_keyframe:
            # Use last keyframe
            self._apply_keyframe(prev_keyframe)
    
    def _interpolate_keyframes(self, prev_keyframe: AnimationKeyframe, next_keyframe: AnimationKeyframe, t: float):
        """Interpolate between two keyframes"""
        # Interpolate blend weights
        for blend_name in self.face_rig.blend_shapes:
            prev_weight = prev_keyframe.blend_weights.get(blend_name, 0.0)
            next_weight = next_keyframe.blend_weights.get(blend_name, 0.0)
            
            interpolated_weight = prev_weight + (next_weight - prev_weight) * t
            self.face_rig.blend_shapes[blend_name].weight = interpolated_weight
    
    def _apply_keyframe(self, keyframe: AnimationKeyframe):
        """Apply single keyframe"""
        for blend_name, weight in keyframe.blend_weights.items():
            if blend_name in self.face_rig.blend_shapes:
                self.face_rig.blend_shapes[blend_name].weight = weight
    
    def render_frame(self):
        """Render a single frame"""
        frame_start_time = time.time()
        
        # Update animation
        self.update_animation()
        
        # Clear buffers
        self.gl_context.clear(0.0, 0.0, 0.0, 1.0)
        self.gl_context.clear(depth=1.0)
        
        # Calculate final mesh with blend shapes
        final_vertices = self._calculate_blended_mesh()
        
        # Update vertex buffer
        if self.vbo:
            self.vbo.write(final_vertices.astype(np.float32).tobytes())
        else:
            self._create_vertex_buffer(final_vertices)
        
        # Set up matrices
        model_matrix = self._get_model_matrix()
        view_matrix = self._get_view_matrix()
        projection_matrix = self._get_projection_matrix()
        
        # Set uniforms
        self.shader_program['model'].write(model_matrix.astype(np.float32).tobytes())
        self.shader_program['view'].write(view_matrix.astype(np.float32).tobytes())
        self.shader_program['projection'].write(projection_matrix.astype(np.float32).tobytes())
        
        # Lighting uniforms
        self.shader_program['lightPos'] = (2.0, 2.0, 2.0)
        self.shader_program['lightColor'] = (1.0, 1.0, 1.0)
        self.shader_program['viewPos'] = (0.0, 0.0, 3.0)
        
        # Render
        if self.vao:
            self.vao.render()
        
        # Swap buffers
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
        # Update performance stats
        frame_time = time.time() - frame_start_time
        self._update_render_stats(frame_time)
    
    def _calculate_blended_mesh(self) -> np.ndarray:
        """Calculate final mesh with blend shapes applied"""
        final_vertices = self.face_rig.base_mesh.copy()
        
        # Apply blend shapes
        for blend_shape in self.face_rig.blend_shapes.values():
            if blend_shape.weight > 0.0:
                # Add weighted blend shape displacement
                displacement = blend_shape.vertices - self.face_rig.base_mesh
                final_vertices += displacement * blend_shape.weight
        
        return final_vertices
    
    def _create_vertex_buffer(self, vertices: np.ndarray):
        """Create vertex buffer object"""
        # Combine vertices with normals and texture coordinates
        normals = self._calculate_normals(vertices)
        
        # Interleave vertex data
        vertex_data = np.zeros((len(vertices), 8), dtype=np.float32)
        vertex_data[:, 0:3] = vertices
        vertex_data[:, 3:6] = normals
        vertex_data[:, 6:8] = self.face_rig.texture_coords
        
        # Create buffer
        self.vbo = self.gl_context.buffer(vertex_data.tobytes())
        
        # Create vertex array object
        self.vao = self.gl_context.vertex_array(self.shader_program, [(self.vbo, '3f 3f 2f', 'aPos', 'aNormal', 'aTexCoord')])
    
    def _calculate_normals(self, vertices: np.ndarray) -> np.ndarray:
        """Calculate vertex normals"""
        normals = np.zeros_like(vertices)
        
        # Calculate face normals and accumulate to vertices
        for face in self.face_rig.face_indices:
            v1, v2, v3 = vertices[face]
            
            # Calculate face normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            face_normal = np.cross(edge1, edge2)
            face_normal = face_normal / np.linalg.norm(face_normal)
            
            # Accumulate to vertex normals
            normals[face] += face_normal
        
        # Normalize vertex normals
        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            if norm > 0:
                normals[i] /= norm
        
        return normals
    
    def _get_model_matrix(self) -> np.ndarray:
        """Get model transformation matrix"""
        # Simple rotation for now
        angle = time.time() * 0.1
        rotation = Rotation.from_euler('y', angle)
        model_matrix = np.eye(4)
        model_matrix[:3, :3] = rotation.as_matrix()
        return model_matrix
    
    def _get_view_matrix(self) -> np.ndarray:
        """Get view transformation matrix"""
        # Simple camera setup
        eye = np.array([0.0, 0.0, 3.0])
        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        
        # Calculate view matrix
        f = center - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        
        u = np.cross(s, f)
        
        view_matrix = np.eye(4)
        view_matrix[0, :3] = s
        view_matrix[1, :3] = u
        view_matrix[2, :3] = -f
        view_matrix[:3, 3] = -eye
        
        return view_matrix
    
    def _get_projection_matrix(self) -> np.ndarray:
        """Get projection transformation matrix"""
        fov = np.radians(45.0)
        aspect = self.width / self.height
        near = 0.1
        far = 100.0
        
        f = 1.0 / np.tan(fov / 2.0)
        
        projection_matrix = np.zeros((4, 4))
        projection_matrix[0, 0] = f / aspect
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (far + near) / (near - far)
        projection_matrix[2, 3] = (2 * far * near) / (near - far)
        projection_matrix[3, 2] = -1.0
        
        return projection_matrix
    
    def _update_render_stats(self, frame_time: float):
        """Update rendering performance statistics"""
        self.render_stats['frames_rendered'] += 1
        
        # Update average frame time
        total_frames = self.render_stats['frames_rendered']
        current_avg = self.render_stats['avg_frame_time']
        self.render_stats['avg_frame_time'] = (current_avg * (total_frames - 1) + frame_time) / total_frames
        
        # Calculate FPS
        if frame_time > 0:
            self.render_stats['fps'] = 1.0 / frame_time
    
    def get_render_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics"""
        return self.render_stats.copy()
    
    def should_close(self) -> bool:
        """Check if window should close"""
        return glfw.window_should_close(self.window)
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up 3D face engine...")
        
        if self.vao:
            self.vao.release()
        if self.vbo:
            self.vbo.release()
        if self.shader_program:
            self.shader_program.release()
        if self.gl_context:
            self.gl_context.release()
        
        if self.window:
            glfw.destroy_window(self.window)
        
        glfw.terminate()
        
        logger.info("✅ 3D face engine cleanup complete")

# Integration class for complete face animation system
class Shock2FaceAnimationSystem:
    """Complete face animation system integrating lip sync and 3D rendering"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.lipsync_engine = LipSyncEngine(config.get('lipsync', {}))
        self.face_engine = Shock3DFaceEngine(config.get('face_3d', {}))
        
        # Animation state
        self.is_speaking = False
        self.current_expression = FaceExpression.NEUTRAL
        
        # Performance tracking
        self.system_stats = {
            'total_animations': 0,
            'avg_sync_accuracy': 0.0
        }
    
    async def speak_with_animation(self, text: str, audio_data: np.ndarray, sample_rate: int, expression: FaceExpression = FaceExpression.NEUTRAL):
        """Perform complete speech animation with lip sync and expression"""
        try:
            self.is_speaking = True
            
            # Set base expression
            self.face_engine.set_expression(expression, intensity=0.8)
            
            # Generate lip sync data
            lipsync_data = await self.lipsync_engine.generate_lipsync_data(audio_data, sample_rate, text)
            
            # Start lip sync animation
            self.lipsync_engine.start_animation(lipsync_data)
            
            # Animation loop
            while self.lipsync_engine.is_animation_active():
                # Get current mouth shape
                mouth_shape = self.lipsync_engine.get_current_mouth_shape()
                
                # Apply to 3D face
                self.face_engine.apply_mouth_shape(mouth_shape)
                
                # Render frame
                self.face_engine.render_frame()
                
                # Check for window close
                if self.face_engine.should_close():
                    break
                
                # Small delay for frame rate control
                await asyncio.sleep(1.0 / 30.0)  # 30 FPS
            
            self.is_speaking = False
            
            # Return to neutral expression
            self.face_engine.set_expression(FaceExpression.NEUTRAL, intensity=1.0)
            
            logger.info("✅ Speech animation completed")
            
        except Exception as e:
            logger.error(f"Speech animation failed: {e}")
            self.is_speaking = False
    
    def set_idle_expression(self, expression: FaceExpression):
        """Set idle expression when not speaking"""
        if not self.is_speaking:
            self.face_engine.set_expression(expression)
            self.current_expression = expression
    
    def render_idle_frame(self):
        """Render single idle frame"""
        if not self.is_speaking:
            self.face_engine.render_frame()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get complete system statistics"""
        return {
            'system': self.system_stats,
            'lipsync': self.lipsync_engine.get_lipsync_stats(),
            'rendering': self.face_engine.get_render_stats()
        }
    
    def cleanup(self):
        """Cleanup all components"""
        self.lipsync_engine.stop_animation()
        self.face_engine.cleanup()
