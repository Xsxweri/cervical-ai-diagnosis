import React, { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import AgentDiagnosis from './components/AgentDiagnosis';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <h1 className="text-2xl font-bold text-gray-900">
              🤖 AI-Agent 宫颈病理诊断系统
            </h1>
            <div className="text-sm text-gray-500">
              基于 YOLOv8 + ResNet50 + LangGraph
            </div>
          </div>
        </header>
        
        <main className="max-w-7xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<AgentDiagnosis />} />
            <Route path="/agent" element={<AgentDiagnosis />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;