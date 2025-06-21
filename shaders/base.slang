/* Copyright (c) 2025, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

struct SceneData
{
    float4x4 projection;
    float4x4 view;
};
ConstantBuffer<SceneData> sceneData;

struct ObjectShaderData {
    float4x4 model;
    float4 color;
};
StructuredBuffer<ObjectShaderData> objectData;

struct VSInput
{
	float3 Pos : POSITION0;
	float3 Normal;
};

struct VSOutput
{
	float4 Pos : SV_POSITION;
    float3 Normal;
    float3 Color;
    float3 ViewVec;
    float3 LightVec;
};

[shader("vertex")]
VSOutput main(VSInput input, uniform uint objIndex)
{
    float4 pos = mul(objectData[objIndex].model, float4(input.Pos, 1.0));

    VSOutput output;
    output.Color = objectData[objIndex].color.rgb;
    output.Normal = mul((float3x3)objectData[objIndex].model, input.Normal);
    output.Pos = mul(sceneData.projection, mul(sceneData.view, pos));

    float3 lightPos = float3(0.0, 0.0, 150.0f);
    output.LightVec = lightPos - pos.xyz;
    output.ViewVec = -pos.xyz;

	return output;
}

[shader("fragment")]
float4 main(VSOutput input)
{
    float3 ambient = input.Color * float3(0.25, 0.25, 0.25);
    float3 N = normalize(input.Normal);
    float3 L = normalize(input.LightVec);
    float3 V = normalize(input.ViewVec);
    float3 R = reflect(-L, N);
    float3 diffuse = max(dot(N, L), 0.0) * input.Color;
    return float4(ambient + diffuse, 1.0);
}