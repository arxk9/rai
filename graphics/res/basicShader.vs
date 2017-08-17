#version 400

attribute vec3 position;
attribute vec2 texCoord;
attribute vec3 normal;
attribute vec3 colorCoord;

varying vec2 texCoord0;
varying vec3 normal0;
varying vec3 position0;
varying vec3 colorCoord0;
varying vec3 viewDirection0;

uniform mat4 MVP;
uniform mat4 Normal;
uniform vec3 cameraPos;
uniform vec3 colorMono;

void main()
{
	gl_Position = MVP * vec4(position, 1.0);
	position0 = gl_Position.xyz;
	texCoord0 = texCoord;
	colorCoord0 = colorMono;
	normal0 = (Normal * vec4(normal, 0.0)).xyz;

	viewDirection0 = vec3(Normal * gl_Position - vec4(cameraPos, 1.0));
}