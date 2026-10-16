"""
The "water eye" floating ball: a GPU shader (GLSL) drawn off-screen into an
image, which FloatingBall then paints. Closed it's a drop of liquid light;
open it's an eye whose pupil follows the gaze point.

The shader is the one from the HTML/WebGL design, unchanged except that
desktop OpenGL doesn't allow `precision` outside OpenGL ES. Off-screen
(not a live OpenGL window) because a see-through, always-on-top OpenGL
window is unreliable on Windows; painting an image keeps the ball's
transparency, dragging and edge docking exactly as before.

    WaterBallRenderer().render(size_px, t, open_, hover, glow, gaze) -> QImage
"""
from __future__ import annotations

import struct

from PySide6.QtCore import QSize
from PySide6.QtGui import QImage, QOffscreenSurface, QOpenGLContext, QVector2D, QVector3D
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLShader,
                              QOpenGLShaderProgram)

GL_FLOAT, GL_TRIANGLE_STRIP, GL_COLOR_BUFFER_BIT = 0x1406, 0x0005, 0x4000

PALETTES = {   # 5 colours each: deep liquid, then 4 the light cycles through
    "Aurora": ["#0a0820", "#3b1e8f", "#6a5cff", "#19b8ff", "#e45cff"],
    "Lagoon": ["#03141a", "#0b5d63", "#19c3a6", "#7cf0ff", "#ffd36b"],
    "Ember":  ["#14060a", "#6b0f3a", "#ff4d6d", "#ff9e3d", "#ffd9a0"],
    "Nebula": ["#0b0618", "#5a189a", "#ff5fa2", "#48cae4", "#b8f2e6"],
}

VERT = "attribute vec2 a; void main(){ gl_Position = vec4(a, 0., 1.); }"

FRAG = """
#ifdef GL_ES
precision highp float;
#endif
uniform vec2 uRes; uniform float uTime, uOpen, uHover, uGlow; uniform vec2 uGaze;
uniform vec3 uP0, uP1, uP2, uP3, uP4;

float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x*p.y); }
float noise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
  return mix(mix(hash(i), hash(i+vec2(1,0)), u.x), mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), u.x), u.y); }
float fbm(vec2 p){ float v = 0.0, a = 0.5; mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
  for (int i = 0; i < 5; i++){ v += a*noise(p); p = m*p; a *= 0.5; } return v; }
float fbm3(vec2 p){ float v = 0.0, a = 0.5; mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
  for (int i = 0; i < 3; i++){ v += a*noise(p); p = m*p; a *= 0.5; } return v/0.875; }
vec3 pal(float t){ t = fract(t)*4.0;
  vec3 c = mix(uP1, uP2, clamp(t, 0.0, 1.0)); c = mix(c, uP3, clamp(t-1.0, 0.0, 1.0));
  c = mix(c, uP4, clamp(t-2.0, 0.0, 1.0)); c = mix(c, uP1, clamp(t-3.0, 0.0, 1.0)); return c; }

void main(){
  float S = min(uRes.x, uRes.y);
  vec2 p = (gl_FragCoord.xy - 0.5*uRes) / (0.5*S);
  float t = uTime;
  float ang = atan(p.y, p.x);
  float o = clamp(uOpen, 0.0, 1.1);

  float wob = 0.010*sin(3.0*ang + t*1.1) + 0.007*sin(5.0*ang - t*1.6) + uHover*0.016*sin(7.0*ang - t*6.0);
  float R = 0.80*(1.0 + wob)*(1.0 + 0.012*sin(t*1.4));
  vec2 u = p / R; float d = length(u);
  float px = 2.0/S/R;
  float mask = 1.0 - smoothstep(1.0 - px, 1.0 + px*0.5, d);
  float g = exp(-max(d - 1.0, 0.0)*9.0)*(1.0 - mask)*uGlow;

  float dd = min(d, 0.999);
  float z = sqrt(1.0 - dd*dd);
  vec3 n = vec3(u, z);
  float fres = pow(1.0 - z, 2.5);

  vec2 q = u*(1.0 + 0.45*(1.0 - z));
  float rot = t*0.05 + o*0.6;
  vec2 w = mat2(cos(rot), -sin(rot), sin(rot), cos(rot))*q*0.95;
  vec2 warp = vec2(fbm3(w + vec2(0.0, t*0.06)), fbm3(w + vec2(4.1, 2.7) - t*0.05));
  float v = fbm3(w + 1.5*warp);
  float cur = smoothstep(0.5, 0.95, fbm3(w*1.3 + vec2(2.0, -4.0) + warp*0.8 - t*0.02));
  vec3 interior = uP0*(0.75 + 0.5*(1.0 - z));
  interior += pal(v*0.6 + 0.1 + t*0.015)*smoothstep(0.35, 0.85, v)*0.22;
  interior += pal(q.y*0.25 + v*0.7 + 0.6 + t*0.02)*cur*0.22;
  for (int i = 0; i < 3; i++) {
    float fi = float(i);
    float ra = fi*2.1 + t*0.04 + o*0.4;
    vec2 rq = mat2(cos(ra), -sin(ra), sin(ra), cos(ra))*q + 0.28*(warp - 0.5);
    float yc = 0.32*sin(rq.x*1.8 + t*0.35 + fi*1.7) + 0.14*sin(rq.x*3.1 - t*0.25 + fi) + (fi - 1.0)*0.22;
    float dy = rq.y - yc;
    float wd = 0.085 + 0.05*sin(rq.x*2.2 + t*0.5 + fi*3.0);
    float body = exp(-dy*dy/(wd*wd));
    float core = exp(-dy*dy/(wd*wd*0.035 + 0.00012));
    float edgeHi = exp(-pow(dy - wd*0.8, 2.0)/(wd*wd*0.05));
    float along = smoothstep(-1.05, -0.25, rq.x)*smoothstep(1.05, 0.2, rq.x);
    vec3 rc = pal(rq.x*0.35 + fi*0.27 + t*0.02);
    interior += (rc*body*0.50 + mix(rc, vec3(1.0), 0.45)*core*0.75 + mix(rc, vec3(1.0), 0.6)*edgeHi*0.35)*along;
  }
  interior *= 0.6 + 0.55*(1.0 - z*0.6);
  interior += uP3*exp(-pow(length(u - vec2(0.38, -0.45)), 2.0)*8.0)*0.30;

  float W = 0.80, A = 0.50*o;
  float hx = max(1.0 - (u.x*u.x)/(W*W), 0.0);
  float ap = A*pow(hx, 1.35) - abs(u.y);
  float apm = smoothstep(0.0, px*1.2, ap)*step(0.001, A)*step(abs(u.x), W);
  vec2 e = u - uGaze*0.16*min(o, 1.0);
  float ir = length(e), ea = atan(e.y, e.x);
  vec2 dir = vec2(cos(ea), sin(ea));
  float irisR = 0.44, pupilR = 0.13 + 0.015*sin(t*0.9);
  float fib  = 0.5 + 0.5*sin(ea*30.0 + 6.0*fbm(dir*3.0 + ir*4.0 + t*0.04));
  fib = 0.55*fib + 0.45*noise(dir*16.0 + vec2(ir*7.0));
  float fib2 = fbm(dir*6.0 + vec2(ir*10.0, 0.0));
  float rr = clamp((ir - pupilR)/(irisR - pupilR), 0.0, 1.0);
  vec3 irisCol = mix(pal(0.55 + 0.35*rr + 0.15*fib2), pal(0.15 + 0.3*rr), 0.35);
  irisCol *= 0.5 + 0.8*fib;
  irisCol *= 1.0 - 0.35*smoothstep(0.55, 0.75, fbm(dir*5.0 + vec2(ir*6.0, 3.0)));
  irisCol += mix(uP4, vec3(1.0), 0.5)*smoothstep(0.35, 0.0, rr)*0.6;
  irisCol *= mix(1.0, 0.25, smoothstep(0.75, 1.0, rr));
  float irisM = 1.0 - smoothstep(irisR - px*1.5, irisR + px, ir);
  float pupM  = 1.0 - smoothstep(pupilR - px*1.5, pupilR + px, ir);
  vec3 eye = mix(interior*1.25 + pal(v)*0.10, irisCol, irisM);
  eye = mix(eye, uP0*0.25, pupM);
  eye += vec3(1.0)*(1.0 - smoothstep(0.035, 0.05, length(e - vec2(-0.09, 0.09))))*0.95;
  eye += vec3(1.0)*(1.0 - smoothstep(0.012, 0.02, length(e - vec2(0.07, -0.06))))*0.6;
  eye *= 0.65 + 0.35*smoothstep(0.0, 0.10, ap);
  vec3 col = mix(interior*(1.0 - 0.35*clamp(o, 0.0, 1.0)), eye, apm);
  float lidLine = exp(-abs(ap)*90.0)*smoothstep(0.0, 0.08, A)*(1.0 - smoothstep(W, W + 0.04, abs(u.x)));
  col += mix(uP3, vec3(1.0), 0.35)*lidLine*0.9;

  vec3 H = normalize(normalize(vec3(-0.5, 0.65, 0.8)) + vec3(0.0, 0.0, 1.0));
  float nh = max(dot(n, H), 0.0);
  col += vec3(1.0)*pow(nh, 400.0)*1.4 + mix(uP2, vec3(1.0), 0.7)*pow(nh, 18.0)*0.10;
  float cres = smoothstep(0.62, 0.92, d)*smoothstep(0.2, 0.9, dot(normalize(u + 1e-4), normalize(vec2(-0.6, 0.8))));
  col += mix(uP2, vec3(1.0), 0.6)*cres*0.25;
  vec3 rimCol = mix(pal(ang/6.2831 + t*0.03), vec3(1.0), 0.45);
  col += rimCol*fres*0.9;
  col += rimCol*smoothstep(0.90, 0.985, d)*(1.0 - smoothstep(0.985, 1.0, d))*0.6;

  col = 1.0 - exp(-col*1.25);
  vec3 glowCol = mix(uP2, uP3, 0.5);
  gl_FragColor = vec4(col*mask + glowCol*g*0.30, mask + g*0.30);
}
"""


def _rgb(hex_colour: str) -> QVector3D:
    return QVector3D(*(int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)))


class WaterBallRenderer:
    """Runs the shader in an off-screen OpenGL context. Raises RuntimeError if
    this computer can't (no OpenGL, or the shader won't compile): the caller
    then keeps the old hand-painted ball."""

    def __init__(self, palette: str = "Aurora") -> None:
        self.palette = [_rgb(c) for c in PALETTES[palette]]
        self._ctx = QOpenGLContext()
        if not self._ctx.create():
            raise RuntimeError("no OpenGL context")
        self._surface = QOffscreenSurface()
        self._surface.setFormat(self._ctx.format())
        self._surface.create()
        if not self._ctx.makeCurrent(self._surface):
            raise RuntimeError("can't use the OpenGL context")
        self._prog = QOpenGLShaderProgram()
        for kind, src in ((QOpenGLShader.ShaderTypeBit.Vertex, VERT), (QOpenGLShader.ShaderTypeBit.Fragment, FRAG)):
            if not self._prog.addShaderFromSourceCode(kind, src):
                raise RuntimeError(f"shader: {self._prog.log()}")
        if not self._prog.link():
            raise RuntimeError(f"shader link: {self._prog.log()}")
        self._quad = QOpenGLBuffer()                     # 2 triangles covering the image
        self._quad.create()
        self._quad.bind()
        data = struct.pack("8f", -1, -1, 1, -1, -1, 1, 1, 1)
        self._quad.allocate(data, len(data))
        self._fbo: QOpenGLFramebufferObject | None = None

    def render(self, size_px: int, t: float, open_: float, hover: float,
               glow: float, gaze: tuple[float, float]) -> QImage:
        """One frame, size_px x size_px. open_ 0 = drop, 1 = eye; gaze in -1..1 (y up)."""
        self._ctx.makeCurrent(self._surface)
        if self._fbo is None or self._fbo.width() != size_px:
            self._fbo = QOpenGLFramebufferObject(QSize(size_px, size_px))
        self._fbo.bind()
        f = self._ctx.functions()
        f.glViewport(0, 0, size_px, size_px)
        f.glClearColor(0, 0, 0, 0)
        f.glClear(GL_COLOR_BUFFER_BIT)
        p = self._prog
        p.bind()
        p.setUniformValue("uRes", QVector2D(size_px, size_px))
        p.setUniformValue1f("uTime", t)
        p.setUniformValue1f("uOpen", open_)
        p.setUniformValue1f("uHover", hover)
        p.setUniformValue1f("uGlow", glow)
        p.setUniformValue("uGaze", QVector2D(*gaze))
        for i, c in enumerate(self.palette):
            p.setUniformValue(f"uP{i}", c)
        self._quad.bind()
        p.enableAttributeArray("a")
        p.setAttributeBuffer("a", GL_FLOAT, 0, 2)
        f.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        image = self._fbo.toImage()
        self._fbo.release()
        return image
