'use client';

import { useState } from 'react';
import { Upload, Palette, Video, Sparkles, Download, Wand2 } from 'lucide-react';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function DrawingAnimationPage() {
  const [drawing, setDrawing] = useState<File | null>(null);
  const [drawingPreview, setDrawingPreview] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState<'cogvideox' | 'svd'>('cogvideox');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [finalVideoUrl, setFinalVideoUrl] = useState<string | null>(null);

  // Handle drawing upload
  const handleDrawingUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setDrawing(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setDrawingPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Generate animation
  const generateAnimation = async () => {
    if (!drawing) {
      toast.error('Please upload a drawing first');
      return;
    }

    setIsGenerating(true);
    setProgress(0);

    try {
      // Upload drawing first
      const formData = new FormData();
      formData.append('file', drawing);
      formData.append('prompt', prompt);
      formData.append('model', model);

      setCurrentStep('Uploading drawing...');
      setProgress(20);

      const response = await axios.post(
        `${API_URL}/api/v1/drawing/animate`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      const jobId = response.data.job_id;

      // Poll for progress
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await axios.get(
            `${API_URL}/api/v1/drawing/status/${jobId}`
          );

          const status = statusResponse.data;
          setProgress(status.progress);
          setCurrentStep(status.current_step);

          if (status.status === 'completed') {
            clearInterval(pollInterval);
            setFinalVideoUrl(status.final_video_url);
            setIsGenerating(false);
            toast.success('Animation generated successfully!');
          } else if (status.status === 'failed') {
            clearInterval(pollInterval);
            setIsGenerating(false);
            toast.error('Animation generation failed');
          }
        } catch (error) {
          console.error('Error polling status:', error);
        }
      }, 2000);

    } catch (error) {
      console.error('Error generating animation:', error);
      toast.error('Failed to generate animation');
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-green-100">
      <Toaster position="top-right" />

      {/* Navigation Header */}
      <header className="border-b border-green-200 bg-white/90 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <Sparkles className="w-8 h-8 text-green-600" />
              <h1 className="text-2xl font-bold text-gray-800">
                AI Video Studio
              </h1>
            </div>

            {/* Navigation Menu */}
            <nav className="flex items-center gap-2">
              <Link href="/">
                <button className="flex items-center gap-2 px-4 py-2 bg-white text-gray-700 border-2 border-green-200 rounded-lg font-semibold hover:bg-green-50 transition">
                  <Video className="w-4 h-4" />
                  Dance Generator
                </button>
              </Link>
              <button className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition">
                <Palette className="w-4 h-4" />
                Drawing Animation
              </button>
            </nav>
          </div>
          <p className="text-gray-600 mt-2 text-sm">
            Bring your drawings to life with CogVideoX & Stable Video Diffusion
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {!finalVideoUrl ? (
          <div className="bg-white shadow-xl rounded-2xl p-8 border border-green-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
              <Palette className="w-6 h-6 text-green-600" />
              Animate Your Drawing
            </h2>

            <div className="space-y-6">
              {/* Upload Area */}
              <div>
                <label className="block text-gray-800 font-semibold mb-3">
                  Upload Drawing
                </label>
                <label className="block cursor-pointer">
                  <div className="border-2 border-dashed border-green-200 rounded-xl p-12 hover:border-green-500 transition text-center bg-green-50/30">
                    {drawingPreview ? (
                      <img
                        src={drawingPreview}
                        alt="Preview"
                        className="max-h-96 mx-auto rounded-lg"
                      />
                    ) : (
                      <div>
                        <Upload className="w-16 h-16 mx-auto text-gray-500 mb-4" />
                        <p className="text-gray-800 text-lg mb-2">
                          Click to upload or drag and drop
                        </p>
                        <p className="text-gray-500">
                          PNG, JPG, sketch, cartoon, or any drawing
                        </p>
                      </div>
                    )}
                  </div>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleDrawingUpload}
                    className="hidden"
                  />
                </label>
              </div>

              {/* Model Selection */}
              <div>
                <label className="block text-gray-800 font-semibold mb-3">
                  AI Model
                </label>
                <div className="grid grid-cols-2 gap-4">
                  <button
                    onClick={() => setModel('cogvideox')}
                    className={`p-4 rounded-xl transition ${
                      model === 'cogvideox'
                        ? 'bg-green-600 text-white'
                        : 'bg-white border-2 border-green-100 text-gray-700 hover:bg-green-50'
                    }`}
                  >
                    <div className="font-semibold">CogVideoX</div>
                    <div className="text-sm opacity-80">
                      Text-guided, high quality
                    </div>
                  </button>
                  <button
                    onClick={() => setModel('svd')}
                    className={`p-4 rounded-xl transition ${
                      model === 'svd'
                        ? 'bg-green-600 text-white'
                        : 'bg-white border-2 border-green-100 text-gray-700 hover:bg-green-50'
                    }`}
                  >
                    <div className="font-semibold">Stable Video Diffusion</div>
                    <div className="text-sm opacity-80">
                      Fast, production-ready
                    </div>
                  </button>
                </div>
              </div>

              {/* Prompt Input */}
              {model === 'cogvideox' && (
                <div>
                  <label className="block text-gray-800 font-semibold mb-2">
                    Animation Prompt (Optional)
                  </label>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="dancing cheerfully, waving hello, running through a field..."
                    rows={3}
                    className="w-full bg-white border-2 border-green-100 rounded-xl px-4 py-3 text-gray-800 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-600"
                  />
                  <p className="text-gray-500 text-sm mt-2">
                    Describe the desired motion or leave blank for automatic animation
                  </p>
                </div>
              )}

              {/* Generate Button */}
              {drawing && (
                <button
                  onClick={generateAnimation}
                  disabled={isGenerating}
                  className="w-full bg-gradient-to-r from-green-600 to-green-600 hover:from-green-700 hover:to-green-700 disabled:from-gray-400 disabled:to-gray-400 text-white font-bold py-4 px-6 rounded-xl transition flex items-center justify-center gap-2 text-lg"
                >
                  {isGenerating ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Generating... {progress}%
                    </>
                  ) : (
                    <>
                      <Wand2 className="w-6 h-6" />
                      Generate Animation
                    </>
                  )}
                </button>
              )}

              {/* Progress */}
              {isGenerating && (
                <div className="space-y-3">
                  <div className="overflow-hidden h-3 rounded-full bg-green-100">
                    <div
                      style={{ width: `${progress}%` }}
                      className="h-full bg-gradient-to-r from-green-600 to-green-600 transition-all duration-500"
                    />
                  </div>
                  <p className="text-center text-gray-600 text-sm">{currentStep}</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="bg-white shadow-xl rounded-2xl p-8 border border-green-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
              Your Animation is Ready!
            </h2>

            <div className="space-y-6">
              <div className="bg-gray-900 rounded-xl overflow-hidden">
                <video
                  src={finalVideoUrl}
                  controls
                  autoPlay
                  loop
                  className="w-full"
                />
              </div>

              <button
                onClick={() => window.open(finalVideoUrl, '_blank')}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-6 rounded-xl transition flex items-center justify-center gap-2"
              >
                <Download className="w-5 h-5" />
                Download Video
              </button>

              <button
                onClick={() => {
                  setDrawing(null);
                  setDrawingPreview(null);
                  setPrompt('');
                  setFinalVideoUrl(null);
                }}
                className="w-full bg-white border-2 border-green-200 hover:bg-green-50 text-gray-700 font-bold py-4 px-6 rounded-xl transition"
              >
                Create Another Animation
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
