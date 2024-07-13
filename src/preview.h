#pragma once

extern GLuint pbo;

std::string currentTimeString();
void initCuda();
bool init();
void mainLoop();
void initPBO();
void initTextures();
bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);