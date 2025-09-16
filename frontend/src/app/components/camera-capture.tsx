"use client"

import { useState, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Camera, X, Check } from "lucide-react"

interface CameraCaptureProps {
  onCapture: (file: File) => void
  onCancel: () => void
}

export default function CameraCapture({ onCapture, onCancel }: CameraCaptureProps) {
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const startCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }, // Use back camera on mobile
      })
      setStream(mediaStream)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
    } catch (error) {
      console.error("Error accessing camera:", error)
      // Fallback to simulated capture for demo
      simulateCapture()
    }
  }, [])

  const simulateCapture = () => {
    const canvas = document.createElement("canvas")
    canvas.width = 640
    canvas.height = 480
    const ctx = canvas.getContext("2d")
    if (ctx) {
      ctx.fillStyle = "#4ade80"
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = "#166534"
      ctx.font = "24px Arial"
      ctx.fillText("Simulated Camera Capture", 150, 240)
      setCapturedImage(canvas.toDataURL())
    }
  }

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current
      const video = videoRef.current
      const ctx = canvas.getContext("2d")

      if (ctx) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx.drawImage(video, 0, 0)
        setCapturedImage(canvas.toDataURL())
      }
    } else {
      // Fallback for demo
      simulateCapture()
    }
  }

  const confirmCapture = () => {
    if (capturedImage) {
      // Convert data URL to File
      fetch(capturedImage)
        .then((res) => res.blob())
        .then((blob) => {
          const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" })
          onCapture(file)
        })
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
    setCapturedImage(null)
    onCancel()
  }

  // Start camera when component mounts
  useState(() => {
    startCamera()
  })

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardContent className="p-4">
        <div className="space-y-4">
          {!capturedImage ? (
            <div className="relative">
              <video ref={videoRef} autoPlay playsInline className="w-full h-64 bg-gray-200 rounded-lg object-cover" />
              <canvas ref={canvasRef} className="hidden" />

              <div className="flex justify-center space-x-4 mt-4">
                <Button onClick={capturePhoto} size="lg">
                  <Camera className="h-5 w-5 mr-2" />
                  Capture
                </Button>
                <Button onClick={stopCamera} variant="outline" size="lg">
                  <X className="h-5 w-5 mr-2" />
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <img
                src={capturedImage || "/placeholder.svg"}
                alt="Captured"
                className="w-full h-64 object-cover rounded-lg"
              />

              <div className="flex justify-center space-x-4">
                <Button onClick={confirmCapture} size="lg">
                  <Check className="h-5 w-5 mr-2" />
                  Use Photo
                </Button>
                <Button onClick={() => setCapturedImage(null)} variant="outline" size="lg">
                  Retake
                </Button>
                <Button onClick={stopCamera} variant="outline" size="lg">
                  <X className="h-5 w-5 mr-2" />
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
