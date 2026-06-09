export default function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-16 border-b border-[#E5E5E5] bg-[#FAFAFA] backdrop-blur">
    <div className="flex h-16 w-full items-center justify-between px-4">
        
        <div className="flex items-center gap-2">
        <div>
            <h1 className="text-lg font-bold tracking-tight text-[#0A0A0A]">
            HPM
            </h1>
            <p className="text-xs text-gray-500">
            회의피하지마
            </p>
        </div>
        </div>

    </div>
    </header>
  );
}