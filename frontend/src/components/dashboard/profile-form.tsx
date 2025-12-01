// 'use client'
// import { useState } from "react"
// import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
// import { Button } from "@/components/ui/button"
// import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
// import { Save, Loader2, Camera, CheckCircle } from "lucide-react"

// export function ProfileForm() {
//   const [isLoading, setIsLoading] = useState(false)
//   const [saveSuccess, setSaveSuccess] = useState(false)
//   const [avatarPreview, setAvatarPreview] = useState("")
//   const [birdName, setBirdName] = useState<string | null>(null)
//   const [selectedFile, setSelectedFile] = useState<File | null>(null);
//   const [outputImage, setOutputImage] = useState<string | null>(null); // <-- thêm

//   const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
//     const file = e.target.files?.[0]
//     if (file) {
//       const reader = new FileReader()
//       reader.onloadend = () => {
//         setAvatarPreview(reader.result as string)
//       }
//       reader.readAsDataURL(file)
//       setSelectedFile(file);
//       setOutputImage(null); // reset ảnh output khi chọn ảnh mới
//       setBirdName(null);
//     }
//   }

//   const handleClassify = async () => {
//     if (!avatarPreview) {
//       alert("Vui lòng tải ảnh con chim trước!")
//       return
//     }
//     const formData = new FormData();
//     if (selectedFile) {
//       formData.append("file", selectedFile);
//     }
//     try {
//       setIsLoading(true)
//       setSaveSuccess(false)
//       setBirdName(null)
//       setOutputImage(null)

//       const response = await fetch('http://localhost:8000/classified', {
//         method: 'POST',
//         body: formData,
//       });
//       const data = await response.json();
//       console.log('Response from server:', data);

//       setBirdName(data.prediction);

//       if (data.file) {
//         setOutputImage(`http://localhost:8000/${data.file}`);
//       }
      
//       setIsLoading(false)
//       setSaveSuccess(true)
//       setTimeout(() => setSaveSuccess(false), 3000)
//     } catch (error) {
//       console.error('Error during classification:', error);
//       setIsLoading(false)
//     }
//   }

//   return (
//     <div className="max-w-4xl space-y-6">
//       {/* Header */}
//       <Card className="border-2 shadow-lg">
//         <CardHeader className="space-y-1">
//           <CardTitle className="text-2xl font-bold">Phân Loại Chim</CardTitle>
//           <CardDescription>
//             Tải ảnh con chim để hệ thống mô phỏng phân loại
//           </CardDescription>
//         </CardHeader>
//       </Card>

//       {/* Bird Image Upload */}
//       <Card className="border-2 shadow-lg">
//         <CardHeader>
//           <CardTitle>Hình Ảnh Con Chim</CardTitle>
//           <CardDescription>
//             Nhấn vào hình để tải ảnh
//           </CardDescription>
//         </CardHeader>

//         <CardContent>
//           <div className="flex items-center gap-6">
//             <div className="relative group">
//               <Avatar className="h-24 w-24 border-4 border-muted">
//                 <AvatarImage src={avatarPreview} alt="Bird image" />
//                 <AvatarFallback className="text-2xl bg-gradient-to-br from-primary/20 to-primary/5">
//                   IMG
//                 </AvatarFallback>
//               </Avatar>

//               <label 
//                 htmlFor="bird-image-upload" 
//                 className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-full opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
//               >
//                 <Camera className="h-6 w-6 text-white" />
//               </label>

//               <input
//                 id="bird-image-upload"
//                 type="file"
//                 accept="image/*"
//                 className="hidden"
//                 onChange={handleAvatarChange}
//               />
//             </div>

//             <div className="flex-1 space-y-2">
//               <p className="text-sm font-medium">
//                 Nhấn vào ảnh để chọn hình con chim cần phân loại
//               </p>
//               <p className="text-xs text-muted-foreground">
//                 Hỗ trợ JPG, PNG, GIF — tối đa 2MB
//               </p>
//             </div>
//           </div>

//           {/* Result */}
//           {birdName && (
//             <div className="mt-4 p-3 border rounded-md bg-green-50 text-green-800">
//               Kết quả phân loại: <span className="font-semibold">{birdName}</span>
//             </div>
//           )}

//           {/* Output Image */}
//           {outputImage && (
//             <div className="mt-4">
//               <p className="text-sm font-medium">Ảnh kết quả phân loại:</p>
//               <img src={outputImage} alt="Kết quả phân loại" className="mt-2 rounded-md border" />
//             </div>
//           )}

//           {/* Submit Button */}
//           <div className="flex items-center gap-3 pt-4">
//             <Button
//               onClick={handleClassify}
//               disabled={isLoading}
//               className="min-w-[140px] h-11 font-medium group"
//             >
//               {isLoading ? (
//                 <>
//                   <Loader2 className="mr-2 h-4 w-4 animate-spin" />
//                   Đang phân loại...
//                 </>
//               ) : saveSuccess ? (
//                 <>
//                   <CheckCircle className="mr-2 h-4 w-4" />
//                   Hoàn tất!
//                 </>
//               ) : (
//                 <>
//                   <Save className="mr-2 h-4 w-4 group-hover:scale-110 transition-transform" />
//                   Phân Loại
//                 </>
//               )}
//             </Button>

//             {saveSuccess && (
//               <p className="text-sm text-green-600 dark:text-green-500 animate-in fade-in-50">
//                 Hệ thống đã trả về kết quả phân loại!
//               </p>
//             )}
//           </div>
//         </CardContent>
//       </Card>
//     </div>
//   )
// }





'use client'
import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Save, Loader2, Camera, CheckCircle } from "lucide-react"

export function ProfileForm() {
  const [isLoading, setIsLoading] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [avatarPreview, setAvatarPreview] = useState("")
  const [birdName, setBirdName] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [outputImage, setOutputImage] = useState<string | null>(null); // <-- thêm

  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setAvatarPreview(reader.result as string)
      }
      reader.readAsDataURL(file)
      setSelectedFile(file);
      setOutputImage(null); // reset ảnh output khi chọn ảnh mới
      setBirdName(null);
    }
  }

  const handleClassify = async () => {
    if (!avatarPreview) {
      alert("Vui lòng tải ảnh con chim trước!")
      return
    }
    const formData = new FormData();
    if (selectedFile) {
      formData.append("file", selectedFile);
    }
    try {
      setIsLoading(true)
      setSaveSuccess(false)
      setBirdName(null)
      setOutputImage(null)

      const response = await fetch('http://localhost:8000/classified', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log('Response from server:', data);

      setBirdName(data.prediction);

      if (data.file) {
        setOutputImage(`http://localhost:8000/${data.file}`);
      }
      
      setIsLoading(false)
      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 3000)
    } catch (error) {
      console.error('Error during classification:', error);
      setIsLoading(false)
    }
  }

  return (
    <div className="max-w-4xl space-y-6">
      {/* Header */}
      <Card className="border-2 shadow-lg">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold">Phân Loại Chim</CardTitle>
          <CardDescription>
            Tải ảnh con chim để hệ thống mô phỏng phân loại
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Bird Image Upload */}
      <Card className="border-2 shadow-lg">
        <CardHeader>
          <CardTitle>Hình Ảnh Con Chim</CardTitle>
          <CardDescription>
            Nhấn vào hình để tải ảnh
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="flex items-center gap-6">
            <div className="relative group">
              <Avatar className="h-24 w-24 border-4 border-muted">
                <AvatarImage src={avatarPreview} alt="Bird image" />
                <AvatarFallback className="text-2xl bg-gradient-to-br from-primary/20 to-primary/5">
                  IMG
                </AvatarFallback>
              </Avatar>

              <label 
                htmlFor="bird-image-upload" 
                className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-full opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
              >
                <Camera className="h-6 w-6 text-white" />
              </label>

              <input
                id="bird-image-upload"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleAvatarChange}
              />
            </div>

            <div className="flex-1 space-y-2">
              <p className="text-sm font-medium">
                Nhấn vào ảnh để chọn hình con chim cần phân loại
              </p>
              <p className="text-xs text-muted-foreground">
                Hỗ trợ JPG, PNG, GIF — tối đa 2MB
              </p>
            </div>
          </div>

          {/* Result */}
          {birdName && (
            <div className="mt-4 p-3 border rounded-md bg-green-50 text-green-800">
              Kết quả phân loại: <span className="font-semibold">{birdName}</span>
            </div>
          )}

          {/* Output Image */}
          {/* {outputImage && (
            <div className="mt-4">
              <p className="text-sm font-medium">Ảnh kết quả phân loại:</p>
              <img src={outputImage} alt="Kết quả phân loại" className="mt-2 rounded-md border" />
            </div>
          )} */}

          {/* Submit Button */}
          <div className="flex items-center gap-3 pt-4">
            <Button
              onClick={handleClassify}
              disabled={isLoading}
              className="min-w-[140px] h-11 font-medium group"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Đang phân loại...
                </>
              ) : saveSuccess ? (
                <>
                  <CheckCircle className="mr-2 h-4 w-4" />
                  Hoàn tất!
                </>
              ) : (
                <>
                  <Save className="mr-2 h-4 w-4 group-hover:scale-110 transition-transform" />
                  Phân Loại
                </>
              )}
            </Button>

            {saveSuccess && (
              <p className="text-sm text-green-600 dark:text-green-500 animate-in fade-in-50">
                Hệ thống đã trả về kết quả phân loại!
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
