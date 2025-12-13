import { useRef } from 'react'

export default function FileUpload({ onUpload }) {
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file && file.type === 'text/csv') {
      onUpload(file)
    } else {
      alert('Please upload a CSV file')
    }
  }

  return (
    <div className="mb-2">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".csv"
        className="hidden"
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        className="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400"
      >
        📁 Upload CSV Data
      </button>
    </div>
  )
}


