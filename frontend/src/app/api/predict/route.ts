import { type NextRequest, NextResponse } from "next/server"
import { MongoClient } from "mongodb"

const MONGODB_URI = process.env.MONGODB_URI || "mongodb://localhost:27017"
const DB_NAME = "tomato_farming"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("image") as File
    const location = formData.get("location") as string

    if (!file) {
      return NextResponse.json({ error: "Please upload a valid image file" }, { status: 400 })
    }

    // Validate file type
    const allowedTypes = ["image/jpeg", "image/jpg", "image/png"]
    if (!allowedTypes.includes(file.type)) {
      return NextResponse.json({ error: "Please upload a valid image file (JPEG, JPG, or PNG)" }, { status: 400 })
    }

    // Validate file size (5MB limit)
    if (file.size > 5 * 1024 * 1024) {
      return NextResponse.json({ error: "Image size should be less than 5MB" }, { status: 400 })
    }

    // Convert file to buffer for processing
    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Here you would call your actual ML model
    // For demo purposes, we'll simulate the prediction
    const mockPrediction = {
      pestName: "Tomato Hornworm",
      confidence: 87.5,
      description:
        "Large green caterpillar that feeds on tomato plants, causing significant damage to leaves and fruit.",
      prevention: [
        "Regular inspection of plants",
        "Hand-picking of caterpillars",
        "Use of beneficial insects like parasitic wasps",
        "Crop rotation to break pest cycles",
      ],
      pesticides: [
        "Bacillus thuringiensis (Bt) - organic option",
        "Spinosad - low toxicity insecticide",
        "Neem oil - natural pesticide",
        "Pyrethrin-based sprays for severe infestations",
      ],
      images: ["/placeholder.svg?height=200&width=200"],
    }

    // Log prediction to MongoDB
    try {
      const client = new MongoClient(MONGODB_URI)
      await client.connect()

      const db = client.db(DB_NAME)
      const collection = db.collection("predictions")

      await collection.insertOne({
        timestamp: new Date(),
        pestName: mockPrediction.pestName,
        confidence: mockPrediction.confidence,
        location: location || "Unknown",
        imageSize: file.size,
        fileName: file.name,
      })

      await client.close()
    } catch (dbError) {
      console.error("MongoDB logging error:", dbError)
      // Continue with response even if logging fails
    }

    return NextResponse.json(mockPrediction)
  } catch (error) {
    console.error("Prediction error:", error)
    return NextResponse.json({ error: "Failed to analyze image. Please try again." }, { status: 500 })
  }
}
