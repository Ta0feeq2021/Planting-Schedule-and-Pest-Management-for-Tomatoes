import { NextResponse } from "next/server"
import { MongoClient } from "mongodb"

export async function GET() {
  try {
    console.log("Testing MongoDB connection...")

    const MONGODB_URI = process.env.MONGODB_URI || "mongodb://localhost:27017/pest"
    console.log("Using MongoDB URI:", MONGODB_URI)

    const client = new MongoClient(MONGODB_URI)

    // Test connection
    await client.connect()
    console.log("Connected to MongoDB successfully!")

    // Test database access
    const db = client.db("pest")
    const collections = await db.listCollections().toArray()
    console.log(
      "Available collections:",
      collections.map((c) => c.name),
    )

    // Test a simple query
    const pestCollection = db.collection("pest_details")
    const pestCount = await pestCollection.countDocuments()

    await client.close()

    return NextResponse.json({
      success: true,
      message: "MongoDB connection successful",
      database: "pest",
      collections: collections.map((c) => c.name),
      pestCount: pestCount,
    })
  } catch (error) {
    console.error("MongoDB connection error:", error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
        troubleshooting: [
          "1. Make sure MongoDB is running: 'mongod' or 'brew services start mongodb-community'",
          "2. Check if port 27017 is available: 'netstat -an | grep 27017'",
          "3. Try connecting with MongoDB Compass to test the connection",
          "4. Verify the database 'pest' exists in your MongoDB instance",
        ],
      },
      { status: 500 },
    )
  }
}
