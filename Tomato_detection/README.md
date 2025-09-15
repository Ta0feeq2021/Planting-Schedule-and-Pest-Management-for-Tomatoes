# Pest Classification and Detection System using Deep Learning

Pests significantly affect agricultural yields, leading to declines in productivity and nutrient depletion. Excessive pesticide usage often results in increased pesticide residues, disrupting the food chain and causing adverse effects on human health and the environment. Our deep learning-based solution automates pest detection and classification to address these challenges.

## Key Features

✅ **Automated Pest Detection** - Advanced deep learning model for accurate pest identification  
⚡ **Efficient Processing** - Rapid image analysis outperforming manual inspection  
🎯 **High Accuracy** - State-of-the-art classification performance  
🌐 **Scalable Solution** - Ready for large-scale agricultural deployment  
🖥️ **User-Friendly Interface** - Accessible to non-technical users  
📚 **Comprehensive Pest Database** - Detailed information including descriptions and treatment recommendations  

## System Screenshots

### Prediction Interface
![Prediction Page](https://github.com/NikhilAMathew/Pest-Classification-and-Detection-System-using-Deep-Learning/assets/117088447/a76d6075-cf97-43d8-887b-d98ddc40794c)

### Results Display
![Result Page](https://github.com/NikhilAMathew/Pest-Classification-and-Detection-System-using-Deep-Learning/assets/117088447/415febc8-9588-4e37-8d57-b134fda46f10)

### Administrative Dashboard
![Officer Dashboard](https://github.com/NikhilAMathew/Pest-Classification-and-Detection-System-using-Deep-Learning/assets/117088447/65110c1b-e7b2-4326-a61a-2eb19c8695fd)

## Technical Implementation

- **Deep Learning Model**: Custom-trained CNN for pest classification
- **Backend**: Python with Flask framework
- **Database**: MongoDB for pest information storage
- **Frontend**: Responsive web interface

## Installation Guide

### Prerequisites
- Python 3.8+
- MongoDB
- pip package manager

### Setup Instructions

1. **Download Required Files**:
   - [Download model and dataset](https://drive.google.com/drive/folders/1Fb1J3y-kVtWnOmW-K21Bb-bIz6wlffdL?usp=sharing)

2. **Prepare Model Files**:
   ```bash
   mkdir models
   mv pest_model.pth models/
   ```

3. **Database Setup**:
   ```bash
   # Create MongoDB database and collection
   mongo
   > use pest
   > db.createCollection("pest_details")
   # Import JSON data (use the downloaded file)
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**:
   ```bash
   python main.py
   ```

6. **Access the System**:
   Open your browser and navigate to `http://localhost:5000`

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   Frontend UI   │ ←→ │  Flask Server   │ ←→ │     MongoDB     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               ↑
                               │
                      ┌─────────────────┐
                      │                 │
                      │  Deep Learning  │
                      │     Model       │
                      │                 │
                      └─────────────────┘
```

## Contributing

We welcome contributions! Please fork the repository and submit pull requests for any enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact on [gmail](mathewnikhil@gmail.com)
