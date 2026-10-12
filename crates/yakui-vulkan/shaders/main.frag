#version 450
#define NO_TEXTURE 4294967295
#define WORKFLOW_MAIN 0
#define WORKFLOW_TEXT 1
#define WORKFLOW_MAIN_PREMULTIPLIED 2

layout (location = 0) in vec4 in_color;
layout (location = 1) in vec2 in_uv;
layout (location = 2) in float in_opacity;
layout (location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D textures[1000];
layout(push_constant) uniform push_constants {
    uint texture_id;
    uint workflow;
};

void main() {
    if (texture_id == NO_TEXTURE) {
        out_color = vec4(in_color.rgb * in_color.a, in_color.a) * in_opacity;
        return;
    } 
    
    if (workflow == WORKFLOW_TEXT) {
        vec4 coverage = texture(textures[texture_id], in_uv);

        if (in_color.a > 0.0) {
            float alpha = max(max(coverage.r, coverage.g), coverage.b) * in_color.a * coverage.a;

            out_color = vec4(in_color.rgb * alpha, alpha);
        } else {
            out_color = coverage;
        }
    } else {
        vec4 user_texture = texture(textures[texture_id], in_uv);
        vec4 color = in_color * user_texture;
        if (workflow == WORKFLOW_MAIN_PREMULTIPLIED) {
            out_color = vec4(color.rgb * in_color.a, color.a);
        } else {
            out_color = vec4(color.rgb * color.a, color.a);
        }
    }
    out_color *= in_opacity;
}
